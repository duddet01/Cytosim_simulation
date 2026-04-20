"""
run_protrusion_v4.py  —  Oelz, Gelfand & Mogilner 2018
=======================================================
Epoch-based membrane protrusion simulation — v4

"""

import os
import re
import subprocess
import warnings

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy.ndimage import gaussian_filter1d
from scipy.ndimage import label as scipy_label

warnings.filterwarnings("ignore", category=UserWarning)


# =============================================================
#  PATHS
# =============================================================

CYTOSIM_BIN = "/mnt/d/cytosim/bin/sim"
REPORT_BIN  = "/mnt/d/cytosim/bin/report"
WORK_DIR    = "/mnt/d/cytosim/epoch_v4"
TEMPLATE    = os.path.join(WORK_DIR, "oelz2018_epoch_v4.cym.template")
EPOCH_DIR   = os.path.join(WORK_DIR, "epoch_files_v4")
OUT_DIR     = os.path.join(WORK_DIR, "output_v4")


# =============================================================
#  PARAMETERS
# =============================================================

# ── Epoch timing ──────────────────────────────────────────────

N_EPOCHS   = 1600
DT_EPOCH   = 1.0
N_STEPS    = 20
NB_FRAMES  = 5

# ── Reporting ─────────────────────────────────────────────────
REPORT_FRAMES    = [2, 3, 4]
SKIP_FIRST_FRAME = True

# ── Motors on/off ─────────────────────────────────────────────
ENABLE_KINESIN          = True
ENABLE_DYNEIN           = False
CARRY_KINESIN_POSITIONS = True   # inject kinesin positions each epoch
CARRY_DYNEIN_POSITIONS  = True   # inject dynein positions from report
                                  # (False = use Python-computed vertices)

# ── Cell geometry ─────────────────────────────────────────────
CELL_R     = 10.0
N_SEGMENTS = 100

# ── Force threshold ───────────────────────────────────────────

FP_MEMBRANE              = 3.0   # pN  raise to narrow protrusion width
FP_PROCESS               = 1.0   # pN  keep low so processes elongate easily
MEMBRANE_STIFFNESS_SLOPE = 0.0   # pN/um  local penalty
PROCESS_STIFFNESS_SLOPE  = 0.0   # pN/um  lower — less resistance in process
TENSION_SLOPE            = 0.0  # pN/um  very low global tension

# ── Process-tip elongation boost ─────────────────────────────
PROCESS_TIP_BOOST = 0.5   # pN  threshold reduction at tip vertex
TIP_BOOST_WIDTH   = 2     # vertices — narrower boost = sharper tip

# ── Process cap ───────────────────────────────────────────────
ENABLE_PROCESS_CAP = True
MAX_PROCESSES      = 5

# ── Protrusion ────────────────────────────────────────────────

STEP               = 0.2  # um per epoch — gradual growth
NEIGHBOUR_FRACTION = 0.2   # low — don't spread activation to neighbours
MAX_EXT_UM         = 8.0
FORCE_SENSITIVITY  = 0.0

# ── Smoothing / filtering ─────────────────────────────────────

SMOOTH_SIGMA          = 0.5   # small - localised force peak → narrow tip
CONFINE_PROXIMITY_UM  = 2.0
RADIAL_FRACTION_MIN   = 0.3 #basically value of cos of angle between force and radial vector allowed
BOUNDARY_SMOOTH_SIGMA = 2.0   # higher — aggressively flatten non-process membrane

# ── Process detection ─────────────────────────────────────────
PROC_THRESH_UM = 1.0   # um above CELL_R to count as a process

# ── Dynein placement (Feature 2) ─────────────────────────────
DYNEIN_DENSITY            = 0.25   # target dyneins / um boundary
MAX_DYNEIN                = 200
MIN_DYNEIN_SPACING_UM     = 1.0    # min arc distance between dyneins
DYNEIN_PROCESS_ENRICHMENT = 3.0    # fold-enrichment inside processes
                                    # (longer process → more enrichment)

# ── Kinesin ───────────────────────────────────────────────────
N_MT      = 150
N_KINESIN = 200

# ── Physics ───────────────────────────────────────────────────
CONFINE_K = 200.0
MT_LENGTH = 5.0

# ── Figures / sampling ────────────────────────────────────────
N_SNAPSHOTS         = 5
FORCE_PLOT_EPOCH    = None
FORCE_PLOT_N_EPOCHS = 5
END_SAMPLE_EVERY    = 20


# =============================================================
#  DERIVED
# =============================================================

def _dt():           return DT_EPOCH / N_STEPS
def _fe(nd):         return (nd-1) if FORCE_PLOT_EPOCH is None \
                            else min(int(FORCE_PLOT_EPOCH), nd-1)

def _report_frames():
    frames = list(REPORT_FRAMES) if REPORT_FRAMES else list(range(NB_FRAMES))
    if SKIP_FIRST_FRAME and 0 in frames:
        frames.remove(0)
    frames = [f for f in frames if 0 <= f < NB_FRAMES]
    return sorted(set(frames)) if frames else [NB_FRAMES-1]


# =============================================================
#  GEOMETRY
# =============================================================

def make_circle(R=CELL_R, n=N_SEGMENTS):
    a = np.linspace(0, 2*np.pi, n, endpoint=False)
    return np.column_stack([R*np.cos(a), R*np.sin(a)])

def polygon_perimeter(verts):
    return float(np.linalg.norm(np.roll(verts,-1,axis=0)-verts, axis=1).sum())

def polygon_area(verts):
    x, y = verts[:,0], verts[:,1]
    return 0.5*abs(np.dot(x,np.roll(y,-1))-np.dot(y,np.roll(x,-1)))

def write_polygon_file(verts, path):
    with open(path,"w") as fh:
        fh.write("% Polygon boundary vertices\n% x    y\n")
        for v in verts:
            fh.write(f"  {v[0]:15.9f}    {v[1]:15.9f}\n")

def _arc_distances(verts):
    segs = np.linalg.norm(np.diff(verts,axis=0,append=verts[:1]),axis=1)
    return np.concatenate([[0.0], np.cumsum(segs)])


# =============================================================
#  PROCESS DETECTION
# =============================================================

def detect_processes(verts):
    r_vals   = np.linalg.norm(verts, axis=1)
    extended = (r_vals > CELL_R + PROC_THRESH_UM).astype(int)
    n        = len(verts)
    if not extended.any():
        return []
    labeled, n_lab = scipy_label(np.concatenate([extended, extended]))
    procs, seen = [], set()
    for lab in range(1, n_lab+1):
        raw = np.where(labeled==lab)[0]
        key = frozenset(int(i%n) for i in raw)
        if key in seen: continue
        seen.add(key)
        idx   = sorted(key)
        tip_i = idx[int(np.argmax(r_vals[idx]))]
        tv    = verts[tip_i]
        procs.append(dict(
            n_verts    = len(idx),
            vertex_ids = idx,
            tip_vertex = tip_i,
            max_ext_um = float(r_vals[tip_i]-CELL_R),
            tip_x      = float(tv[0]),
            tip_y      = float(tv[1]),
            tip_r      = float(r_vals[tip_i]),
            tip_theta  = float((np.degrees(np.arctan2(tv[1],tv[0]))+360)%360),
        ))
    procs.sort(key=lambda p: p['max_ext_um'], reverse=True)
    for k,p in enumerate(procs): p['label'] = k+1
    return procs

def _process_vertex_set(procs):
    s = set()
    for p in procs: s.update(p['vertex_ids'])
    return s

def print_processes(procs):
    if not procs:
        print(f"    [processes] none  (thresh={PROC_THRESH_UM} um)"); return
    print(f"    [processes] {len(procs)}:")
    print(f"    {'#':>3}  {'ext(um)':>8}  {'tip_x':>8}  "
          f"{'tip_y':>8}  {'theta':>8}  {'n_v':>4}")
    for p in procs:
        print(f"    {p['label']:>3}  {p['max_ext_um']:>8.3f}  "
              f"{p['tip_x']:>8.3f}  {p['tip_y']:>8.3f}  "
              f"{p['tip_theta']:>8.2f}  {p['n_verts']:>4}")


# =============================================================
#  DYNEIN PLACEMENT
# =============================================================

def _target_n_dynein(verts):
    if not ENABLE_DYNEIN: return 0
    return max(1, min(int(round(DYNEIN_DENSITY*polygon_perimeter(verts))),
                      MAX_DYNEIN))

def _place_dyneins_initial(verts):
    """Epoch 0: evenly spaced."""
    n_target  = _target_n_dynein(verts)
    arc_dists = _arc_distances(verts)
    total_arc = arc_dists[-1]
    spacing   = max(total_arc/max(n_target,1), MIN_DYNEIN_SPACING_UM)
    indices, next_arc = [], 0.0
    for i in range(len(verts)):
        if arc_dists[i] >= next_arc - 1e-9:
            indices.append(i)
            next_arc = arc_dists[i] + spacing
            if len(indices) >= n_target: break
    return indices

def _update_dynein_positions(verts, prev_indices, procs):
    n         = len(verts)
    n_target  = _target_n_dynein(verts)
    arc_dists = _arc_distances(verts)
    total_arc = arc_dists[-1]

     remapped = sorted(set(min(int(i), n-1) for i in prev_indices))

    # Gap-filling with process enrichment
    ideal_spacing = total_arc / max(n_target, 1)
    min_gap       = max(ideal_spacing, MIN_DYNEIN_SPACING_UM)
    dyn_arcs      = sorted(arc_dists[i] for i in remapped)
    new_indices   = list(remapped)

    if len(new_indices) < n_target and dyn_arcs:
        # Build per-vertex enrichment weights
        r_verts    = np.linalg.norm(verts, axis=1)
        ext_verts  = np.maximum(0.0, r_verts - CELL_R)   # extension per vertex
        proc_verts = _process_vertex_set(procs)

        # Weight = enrichment_factor × extension for process verts, 1.0 elsewhere
        # This ensures longer process regions attract more dyneins
        weights = np.ones(n)
        for vi in proc_verts:
            weights[vi] = 1.0 + (DYNEIN_PROCESS_ENRICHMENT - 1.0) * (
                1.0 + ext_verts[vi])   # scale with extension

        # Find gaps, fill with weighted random choice
        gaps = []
        for k in range(len(dyn_arcs)):
            a0 = dyn_arcs[k]
            a1 = dyn_arcs[(k+1) % len(dyn_arcs)]
            if k == len(dyn_arcs)-1: a1 += total_arc
            gap_len = a1 - a0
            if gap_len > min_gap * 1.5:
                gaps.append((gap_len, a0, a1 % total_arc))
        gaps.sort(reverse=True)

        for gap_len, a0, a1 in gaps:
            if len(new_indices) >= n_target: break
            # candidates in this arc range
            cands = [i for i in range(n)
                     if a0 <= arc_dists[i] <= a1
                     and i not in new_indices]
            if not cands:
                mid = (a0 + gap_len/2) % total_arc
                bi  = int(np.argmin(np.abs(arc_dists[:-1]-mid)))
                if bi not in new_indices:
                    new_indices.append(bi)
                continue
            w = weights[cands]
            w /= w.sum()
            chosen = int(np.random.choice(cands, p=w))
            new_indices.append(chosen)

    return sorted(new_indices[:n_target])

def build_dynein_block(verts, dynein_indices):
    """Build DYNEIN_BLOCK from Python-computed vertex positions."""
    if not ENABLE_DYNEIN or not dynein_indices: return ""
    lines = []
    for idx in dynein_indices:
        x, y = verts[idx]
        lines += ["new dynein_cortex","{",
                  f"    position = {x:.6f}  {y:.6f}","}"]
    return "\n".join(lines)

def build_dynein_block_from_xy(xy_list):
    """Build DYNEIN_BLOCK from Cytosim-reported (x,y) positions."""
    if not ENABLE_DYNEIN or not xy_list: return ""
    lines = []
    for x, y in xy_list:
        lines += ["new dynein_cortex","{",
                  f"    position = {x:.6f}  {y:.6f}","}"]
    return "\n".join(lines)


# =============================================================
#  MOTOR POSITION PARSING
# =============================================================

def parse_dynein_positions(path):
    """
    Parse dynein_cortex:position report.
    Format:  class  identity  posX  posY  fiber  abscissa
    Returns list of (x, y) from the LAST frame.
    """
    return _parse_xy_report(path, x_col=2, y_col=3)

def parse_kinesin_positions(path):
    """
    Parse kinesin1:state report.
    Format:  class  identity  active  posX  posY  fiber1 abs1 fiber2 abs2
    Returns list of (x, y) from the LAST frame.
    """
    return _parse_xy_report(path, x_col=3, y_col=4)

def _parse_xy_report(path, x_col, y_col):
    """Generic parser: reads LAST frame of any Cytosim report.

    Dynein format:  class  identity  posX  posY  fiber  abscissa
                    col0   col1      col2  col3  col4   col5
    Kinesin format: class  identity  active  posX  posY  ...
                    col0   col1      col2    col3  col4

    All entries in the last frame are returned, including unbound
    motors (fiber=0, abscissa=0) — the anchor position is still valid.
    """
    if not os.path.exists(path): return []
    frames, cur_frame, recs = {}, 0, []

    with open(path) as fh:
        for line in fh:
            s = line.strip()
            if not s: continue
            if s.startswith('% frame'):
                # Save previous frame before starting new one
                if recs:
                    frames[cur_frame] = list(recs)
                recs = []
                try: cur_frame = int(s.split()[-1])
                except: cur_frame = 0
                continue
            if s.startswith('%'): continue
            parts = s.split()
            if len(parts) <= max(x_col, y_col): continue
            try:
                recs.append((float(parts[x_col]), float(parts[y_col])))
            except (ValueError, IndexError):
                continue

    # Save the final frame — this was the bug: last frame never saved
    # because there is no subsequent '% frame' line to trigger the save
    if recs:
        frames[cur_frame] = list(recs)

    if not frames: return []
    return frames[max(frames.keys())]

    _save()
    if not frames: return []
    return frames[max(frames.keys())]


# =============================================================
#  KINESIN BLOCK
# =============================================================

def build_kinesin_block(xy_list):
    """Inject kinesin1 couples at their last-epoch positions."""
    if not ENABLE_KINESIN or not xy_list: return ""
    lines = []
    for x, y in xy_list:
        lines += ["new kinesin1","{",
                  f"    position = {x:.6f}  {y:.6f}","}"]
    return "\n".join(lines)


# =============================================================
#  MT POSITION PARSING
# =============================================================

def parse_fiber_points(path):
    if not os.path.exists(path): return {}
    frames, cur_frame, cur_fiber, cur_pts = {}, None, None, {}
    with open(path) as fh:
        for line in fh:
            s = line.strip()
            if not s: continue
            if s.startswith('% frame'):
                if cur_frame is not None: frames[cur_frame] = cur_pts
                try: cur_frame = int(s.split()[-1])
                except: cur_frame = 0
                cur_pts = {}; cur_fiber = None; continue
            if re.match(r'%\s+fiber\s+f', s):
                m = re.search(r'f\d+:(\d+)', s)
                if m: cur_fiber = int(m.group(1)); cur_pts[cur_fiber] = []
                continue
            if s.startswith('%'): continue
            parts = s.split()
            if len(parts) < 3: continue
            try:
                x, y = float(parts[1]), float(parts[2])
                if cur_fiber is not None: cur_pts[cur_fiber].append((x,y))
            except: continue
    if cur_frame is not None: frames[cur_frame] = cur_pts
    if not frames: return {}
    return frames[max(frames.keys())]

def build_mt_shape_block(fiber_points):
    if not fiber_points: return ""
    lines = []
    for fid, pts in sorted(fiber_points.items()):
        if len(pts) < 2: continue
        ref_x, ref_y = pts[0]
        shape_pts = [f"{px-ref_x:.6f} {py-ref_y:.6f}" for px,py in pts]
        lines += ["new microtubule","{",
                  f" position = {ref_x:.6f}  {ref_y:.6f}",
                  " placement = anywhere"," orientation = off",
                  f" shape = {','.join(shape_pts)};","}"]
    return "\n".join(lines)


# =============================================================
#  WRITE .CYM
# =============================================================

def _replace_bare_token(text, token, content):
    lines, new_lines, replaced = text.split('\n'), [], False
    for line in lines:
        if not replaced and line.strip() == token:
            new_lines.append(content); replaced = True
        else:
            new_lines.append(line)
    return '\n'.join(new_lines) if replaced \
           else text.replace(token, content, 1)

def write_cym(epoch, verts, dynein_indices,
              prev_fiber_points=None,
              prev_dynein_xy=None,
              prev_kinesin_xy=None):
    ep_dir = os.path.join(EPOCH_DIR, f"epoch_{epoch:04d}")
    os.makedirs(ep_dir, exist_ok=True)
    poly = f"polygon_{epoch:04d}.txt"
    write_polygon_file(verts, os.path.join(ep_dir, poly))

    if not os.path.isfile(TEMPLATE):
        raise FileNotFoundError(f"Template not found: {TEMPLATE}")
    with open(TEMPLATE) as fh:
        cym = fh.read()

    # ── MT shapes ────────────────────────────────────────────────────
    mt_shapes = (build_mt_shape_block(prev_fiber_points)
                 if prev_fiber_points and epoch > 0 else "")
    n_mt_cym  = 0 if mt_shapes else N_MT

    # ── Dynein block ─────────────────────────────────────────────────
        if ENABLE_DYNEIN:
        if CARRY_DYNEIN_POSITIONS and prev_dynein_xy and epoch > 0:
            n_report  = len(prev_dynein_xy)
            n_target  = len(dynein_indices)
            if n_report >= n_target:
                # Report has enough — use as-is
                xy_final = prev_dynein_xy[:n_target]
            else:
                # Pad: keep reported positions, fill gaps with Python vertices
                xy_final = list(prev_dynein_xy)
                # Use dynein_indices that are NOT already close to a reported pos
                used_verts = set()
                for rx, ry in prev_dynein_xy:
                    dists = np.linalg.norm(verts - np.array([rx, ry]), axis=1)
                    used_verts.add(int(np.argmin(dists)))
                for vi in dynein_indices:
                    if len(xy_final) >= n_target:
                        break
                    if vi not in used_verts:
                        xy_final.append(tuple(verts[vi]))
                        used_verts.add(vi)
            dynein_block = build_dynein_block_from_xy(xy_final)
            print(f"    [dynein carry] {n_report} from report + "
                  f"{len(xy_final)-n_report} Python-padded = "
                  f"{len(xy_final)} total")
        else:
            dynein_block = build_dynein_block(verts, dynein_indices)
    else:
        dynein_block = ""

    # ── Kinesin block ────────────────────────────────────────────────
    if ENABLE_KINESIN and CARRY_KINESIN_POSITIONS and prev_kinesin_xy and epoch > 0:
        kinesin_block = build_kinesin_block(prev_kinesin_xy)
        n_kin_cym     = 0   # explicit blocks handle placement
        print(f"    [kinesin carry] {len(prev_kinesin_xy)} positions "
              f"from Cytosim report")
    else:
        kinesin_block = ""
        n_kin_cym     = N_KINESIN if ENABLE_KINESIN else 0

    # ── Scalar substitutions ─────────────────────────────────────────
    for tag, val in {
        "EPOCH_IDX":    str(epoch),
        "POLY_FILE":    poly,
        "N_MT":         str(n_mt_cym),
        "N_KINESIN":    str(n_kin_cym),
        "CYM_DT":       f"{_dt():.6f}",
        "CYM_NSTEPS":   str(N_STEPS),
        "CYM_NBFRAMES": str(NB_FRAMES),
    }.items():
        cym = re.sub(r'\b' + re.escape(tag) + r'\b', val, cym)

    # ── Block substitutions ───────────────────────────────────────────
    cym = _replace_bare_token(cym, 'MT_SHAPES_BLOCK', mt_shapes)
    cym = _replace_bare_token(cym, 'DYNEIN_BLOCK',    dynein_block)
    cym = _replace_bare_token(cym, 'KINESIN_BLOCK',   kinesin_block)

    cym_path = os.path.join(ep_dir, f"epoch_{epoch:04d}.cym")
    with open(cym_path,"w") as fh: fh.write(cym)
    return cym_path, ep_dir


# =============================================================
#  RUN CYTOSIM
# =============================================================

def run_cytosim(cym_path, ep_dir):
    r = subprocess.run([CYTOSIM_BIN, os.path.basename(cym_path)],
                       cwd=ep_dir, capture_output=True, text=True, timeout=600)
    if r.returncode != 0:
        print(f"    [!] Cytosim exit {r.returncode}: {r.stderr[-600:]}")
        return False
    return True


# =============================================================
#  REPORTS
# =============================================================

def run_report(ep_dir, epoch):
    rf    = _report_frames()
    last  = [NB_FRAMES-1]
    paths = {
        'confine': os.path.join(ep_dir, f"fiber_confine_{epoch}.txt"),
        'points':  os.path.join(ep_dir, f"fiber_points_{epoch}.txt"),
        'ends':    os.path.join(ep_dir, f"fiber_ends_{epoch}.txt"),
        'dynein':  os.path.join(ep_dir, f"dynein_pos_{epoch}.txt"),
        'kinesin': os.path.join(ep_dir, f"kinesin_pos_{epoch}.txt"),
    }

    def _rep(rtype, out_path, frames=None):
        combined = []
        for fidx in (frames or rf):
            res = subprocess.run([REPORT_BIN, rtype, f"frame={fidx}"],
                                 cwd=ep_dir, capture_output=True,
                                 text=True, timeout=120)
            if res.returncode == 0 and res.stdout.strip():
                combined.append(res.stdout)
            else:
                print(f"    [report] {rtype} f={fidx}: {res.stderr[:100]}")
        with open(out_path,"w") as fh: fh.write("\n".join(combined))

    print(f"    [report] frames={rf}  confine_force ...")
    _rep("fiber:confine_force", paths['confine'])
    print(f"    [report] fiber:point ...")
    _rep("fiber:point",         paths['points'])
    print(f"    [report] fiber:ends ...")
    _rep("fiber:ends",          paths['ends'])
    print(f"    [report] dynein_cortex:position ...")
    _rep("dynein_cortex:position", paths['dynein'],  frames=last)
    print(f"    [report] kinesin1:state ...")
    _rep("kinesin1:state",         paths['kinesin'], frames=last)
    return paths


# =============================================================
#  PARSE FORCES
# =============================================================

def parse_multiframe_confine(path):
    if not os.path.exists(path): return {}
    frames, cur_frame, recs = {}, 0, []

    def _save(fr, r):
        frames[fr] = pd.DataFrame(r) if r else pd.DataFrame()

    with open(path) as fh:
        for line in fh:
            s = line.strip()
            if not s: continue
            if s.startswith('% frame'):
                _save(cur_frame, recs)
                try: cur_frame = int(s.split()[-1])
                except: cur_frame = 0
                recs = []; continue
            if s.startswith('%'): continue
            parts = s.split()
            if len(parts) < 6: continue
            try:
                x,  y  = float(parts[2]), float(parts[3])
                fx, fy = float(parts[4]), float(parts[5])
                r      = np.sqrt(x**2+y**2)+1e-12
                if abs(r-CELL_R) > CONFINE_PROXIMITY_UM: continue
                fr   = -(x*fx+y*fy)/r
                fmag = np.sqrt(fx**2+fy**2)+1e-12
                if fr <= 0: continue
                if fr/fmag < RADIAL_FRACTION_MIN: continue
                recs.append(dict(x=x,y=y,r=r,fr=fr,fmag=fmag))
            except: continue
    _save(cur_frame, recs)
    return frames

def load_forces(ep_dir, epoch):
    path = os.path.join(ep_dir, f"fiber_confine_{epoch}.txt")
    rf   = _report_frames()
    if not os.path.exists(path):
        print(f"    [confine] NOT FOUND: {path}"); return pd.DataFrame()
    print(f"    [confine] {os.path.getsize(path)} bytes  frames={rf}")
    fd = parse_multiframe_confine(path)
    if not fd: return pd.DataFrame()
    ne = {k:v for k,v in fd.items() if k in rf and not v.empty}
    if not ne: ne = {k:v for k,v in fd.items() if not v.empty}
    if not ne: return pd.DataFrame()
    nd  = max(len(ne),1)
    com = pd.concat(list(ne.values()), ignore_index=True)
    com['xr'] = (com['x']/0.05).round()*0.05
    com['yr'] = (com['y']/0.05).round()*0.05
    s = (com.groupby(['xr','yr'],as_index=False)
         .agg(x=('x','mean'),y=('y','mean'),fr=('fr','sum'),fmag=('fmag','sum')))
    s['fr'] /= nd; s['fmag'] /= nd
    print(f"    [confine] denom={nd}  avg_max={s['fr'].max():.3f} pN")
    return s[['x','y','fr','fmag']]


# =============================================================
#  FIBER:ENDS
# =============================================================

def parse_fiber_ends(path):
    if not os.path.exists(path): return []
    recs = []
    with open(path) as fh:
        for line in fh:
            s = line.strip()
            if not s or s.startswith('%'): continue
            parts = s.split()
            if len(parts) < 13: continue
            try:
                recs.append(dict(
                    identity=int(parts[1]),  length=float(parts[2]),
                    stateP=int(parts[3]),
                    posPX=float(parts[4]),   posPY=float(parts[5]),
                    stateM=int(parts[8]),
                    posMX=float(parts[9]),   posMY=float(parts[10]),
                ))
            except: continue
    return recs

def classify_ends_at_tips(ends_recs, procs):
    if not ends_recs: return 0,0
    sectors = []
    for p in procs:
        if p['max_ext_um'] <= 0: continue
        t  = np.radians(p['tip_theta'])
        ha = np.pi*p['n_verts']/N_SEGMENTS + np.pi*1.5/N_SEGMENTS
        sectors.append((t,ha))
    def _in(x,y):
        r = np.sqrt(x**2+y**2)
        if r <= CELL_R: return False
        th = np.arctan2(y,x)
        for t,ha in sectors:
            if abs((th-t+np.pi)%(2*np.pi)-np.pi) <= ha: return True
        return False
    np_out = nm_out = 0
    for rec in ends_recs:
        if not _in(rec['posPX'],rec['posPY']) and \
           not _in(rec['posMX'],rec['posMY']): continue
        rP = np.sqrt(rec['posPX']**2+rec['posPY']**2)
        rM = np.sqrt(rec['posMX']**2+rec['posMY']**2)
        if rP >= rM: np_out += 1
        else:        nm_out += 1
    return np_out, nm_out


# =============================================================
#  SYNTHETIC FALLBACK
# =============================================================

def synthetic_forces(verts, epoch):
    np.random.seed(epoch*17+3)
    hot  = [8,24,40,55,70,88]
    recs = []
    for i,v in enumerate(verts):
        f = np.random.exponential(0.25)
        if i in hot: f += min(1.0+0.08*epoch, 9.0)
        if f >= 0.05: recs.append(dict(x=v[0],y=v[1],fr=f,fmag=f))
    return pd.DataFrame(recs)


# =============================================================
#  PROTRUSION  (with tip boost + process cap)
# =============================================================

def apply_protrusion(verts, forces_df, procs):
    """
    A: bin forces onto nearest vertex
    B: Gaussian smooth seg_force
    C: compute threshold
       + tip boost: reduce threshold near process tips
    D: protrusion + neighbour spread
       + process cap: block non-process vertices when at capacity
    E: boundary shape smoothing
    """
    n         = len(verts)
    new_verts = verts.copy()
    seg_force = np.zeros(n)
    r_verts   = np.linalg.norm(verts, axis=1)

    # A
    if not forces_df.empty:
        for _, row in forces_df.iterrows():
            pt    = np.array([row['x'],row['y']])
            dists = np.linalg.norm(verts-pt, axis=1)
            idx   = int(np.argmin(dists))
            if dists[idx] < 1.5:
                seg_force[idx] += row['fr']

    # B
    seg_force = gaussian_filter1d(seg_force, sigma=SMOOTH_SIGMA, mode='wrap')

    # C: base threshold
    extension    = np.maximum(0.0, r_verts-CELL_R)
    is_process   = extension > PROC_THRESH_UM
    total_excess = float(extension.sum())
    base_thresh  = np.where(is_process, FP_PROCESS, FP_MEMBRANE)
    local_slope  = np.where(is_process,
                            PROCESS_STIFFNESS_SLOPE, MEMBRANE_STIFFNESS_SLOPE)
    thresh = base_thresh + local_slope*extension + TENSION_SLOPE*total_excess

    # C: tip boost — reduce threshold near tips of existing processes
    if PROCESS_TIP_BOOST > 0 and procs:
        boost = np.zeros(n)
        for p in procs:
            tip_v = p['tip_vertex']
            for offset in range(-TIP_BOOST_WIDTH, TIP_BOOST_WIDTH+1):
                vi = (tip_v+offset) % n
                w  = np.exp(-0.5*(offset/max(TIP_BOOST_WIDTH,1))**2)
                boost[vi] = max(boost[vi], PROCESS_TIP_BOOST*w)
        thresh = np.maximum(0.0, thresh-boost)

    # D: activation mask
    active_mask = seg_force > thresh

    # D: process cap — block nucleation when cap is reached
    if ENABLE_PROCESS_CAP and MAX_PROCESSES > 0 \
            and len(procs) >= MAX_PROCESSES:
        proc_v = _process_vertex_set(procs)
        for i in range(n):
            if i not in proc_v:
                active_mask[i] = False

    n_active = int(active_mask.sum())

    # D: protrusion
    disp = np.zeros(n)
    for i in np.where(active_mask)[0]:
        peak = STEP*(1.0+FORCE_SENSITIVITY*(seg_force[i]-thresh[i]))
        if peak > disp[i]: disp[i] = peak
        nb = peak*NEIGHBOUR_FRACTION
        im1=(i-1)%n; ip1=(i+1)%n
        if nb > disp[im1]: disp[im1] = nb
        if nb > disp[ip1]: disp[ip1] = nb

    for j in range(n):
        if disp[j] <= 0: continue
        exc = r_verts[j]-CELL_R
        if exc >= MAX_EXT_UM: continue
        d = min(disp[j], MAX_EXT_UM-exc)
        new_verts[j] += d*(verts[j]/(r_verts[j]+1e-12))

    # E: boundary shape smoothing
    # Only smooth the MEMBRANE (non-process) vertices.
    # Process vertices keep their radii exactly — we don't want to
    # broaden or shrink the protrusion tip.
    # Strategy: smooth the full extension array, then restore
    # process-vertex radii to their unsmoothed values so the tip
    # stays sharp while jagged membrane artefacts are removed.
    if BOUNDARY_SMOOTH_SIGMA > 0:
        r_new     = np.linalg.norm(new_verts, axis=1)
        ext_new   = np.maximum(0.0, r_new - CELL_R)
        ext_smooth = gaussian_filter1d(ext_new,
                                       sigma=BOUNDARY_SMOOTH_SIGMA,
                                       mode='wrap')
        # Restore process vertices to unsmoothed value
        ext_final = ext_smooth.copy()
        proc_ext_thresh = PROC_THRESH_UM * 0.5   # restore above half-threshold
        ext_final[ext_new > proc_ext_thresh] = ext_new[ext_new > proc_ext_thresh]

        angles    = np.arctan2(new_verts[:,1], new_verts[:,0])
        new_verts = np.column_stack(
            [(CELL_R + ext_final)*np.cos(angles),
             (CELL_R + ext_final)*np.sin(angles)])

    return new_verts, seg_force, thresh, n_active


# =============================================================
#  SAVE CSVs
# =============================================================

def save_outputs(b_hist, f_hist, proc_records, dyn_idx_hist,
                 perim_hist, area_hist, end_hist):
    os.makedirs(OUT_DIR, exist_ok=True)

    # Boundary
    rows = []
    for ep,v in enumerate(b_hist):
        for si,pt in enumerate(v):
            r  = float(np.linalg.norm(pt))
            th = (np.degrees(np.arctan2(pt[1],pt[0]))+360)%360
            rows.append(dict(epoch=ep,time_s=round(ep*DT_EPOCH,1),
                seg=si,x=round(pt[0],4),y=round(pt[1],4),
                r=round(r,4),theta=round(th,2)))
    pd.DataFrame(rows).to_csv(f"{OUT_DIR}/boundary_evolution_v4.csv",index=False)
    print(f"[OK] boundary_evolution_v4.csv")

    # Forces
    frows = []
    for ep,f in enumerate(f_hist):
        angs = np.linspace(0,360,len(f),endpoint=False)
        for si,(ang,fv) in enumerate(zip(angs,f)):
            frows.append(dict(epoch=ep,time_s=round(ep*DT_EPOCH,1),
                seg=si,angle_deg=round(ang,2),force_pN=round(float(fv),5)))
    pd.DataFrame(frows).to_csv(f"{OUT_DIR}/membrane_forces_v4.csv",index=False)
    print(f"[OK] membrane_forces_v4.csv")

    # Processes
    prows = []
    for ep,procs in enumerate(proc_records):
        base = dict(epoch=ep,time_s=round(ep*DT_EPOCH,1),
                    time_min=round(ep*DT_EPOCH/60,3),
                    n_procs=len(procs),
                    n_dynein=len(dyn_idx_hist[ep]) if ep<len(dyn_idx_hist) else 0,
                    perimeter_um=round(perim_hist[ep],3),
                    area_um2=round(area_hist[ep],3))
        if not procs:
            prows.append({**base,'label':0,'max_ext_um':0,
                'tip_x':0,'tip_y':0,'tip_r':CELL_R,'tip_theta':0,'n_verts':0})
        else:
            for p in procs:
                prows.append({**base,'label':p['label'],
                    'max_ext_um':round(p['max_ext_um'],4),
                    'tip_x':round(p['tip_x'],4),'tip_y':round(p['tip_y'],4),
                    'tip_r':round(p['tip_r'],4),'tip_theta':round(p['tip_theta'],2),
                    'n_verts':p['n_verts']})
    pd.DataFrame(prows).to_csv(f"{OUT_DIR}/process_growth_v4.csv",index=False)
    print(f"[OK] process_growth_v4.csv")

    # Perimeter/area
    pd.DataFrame([dict(epoch=ep,time_s=round(ep*DT_EPOCH,1),
                       time_min=round(ep*DT_EPOCH/60,3),
                       perimeter_um=round(p,3),area_um2=round(a,3))
                  for ep,(p,a) in enumerate(zip(perim_hist,area_hist))])\
      .to_csv(f"{OUT_DIR}/perimeter_area_v4.csv",index=False)
    print(f"[OK] perimeter_area_v4.csv")

    # Dynein positions
    drows = []
    for ep,indices in enumerate(dyn_idx_hist):
        verts = b_hist[ep] if ep<len(b_hist) else b_hist[-1]
        for rank,vi in enumerate(indices):
            x,y = verts[vi]
            drows.append(dict(epoch=ep,dynein_id=rank,vertex_idx=vi,
                x=round(x,4),y=round(y,4)))
    pd.DataFrame(drows).to_csv(f"{OUT_DIR}/dynein_positions_v4.csv",index=False)
    print(f"[OK] dynein_positions_v4.csv")

    # End polarity
    erows = []
    for entry in end_hist:
        if entry is None: continue
        ep_idx,np_out,nm_out,total = entry
        erows.append(dict(epoch=ep_idx,
            time_s=round(ep_idx*DT_EPOCH,1),
            time_min=round(ep_idx*DT_EPOCH/60,3),
            n_plus_outer=np_out,n_minus_outer=nm_out,n_process_mts=total))
    pd.DataFrame(erows).to_csv(f"{OUT_DIR}/end_polarity_v4.csv",index=False)
    print(f"[OK] end_polarity_v4.csv")

    # Summary
    max_ext   = max((p['max_ext_um'] for pr in proc_records for p in pr),default=0)
    max_nproc = max((len(pr) for pr in proc_records),default=0)
    cap_str   = f"ON  MAX={MAX_PROCESSES}" if ENABLE_PROCESS_CAP else "OFF"
    lines = ["="*66,"  OELZ 2018 — PROTRUSION v4  SUMMARY","="*66,
        f"  CARRY_DYNEIN     : {CARRY_DYNEIN_POSITIONS}",
        f"  CARRY_KINESIN    : {CARRY_KINESIN_POSITIONS}",
        f"  DYNEIN_ENRICHMENT: {DYNEIN_PROCESS_ENRICHMENT}x  "
        f"(longer processes attract more)",
        f"  TIP_BOOST        : {PROCESS_TIP_BOOST} pN  "
        f"width={TIP_BOOST_WIDTH} vertices",
        f"  PROCESS_CAP      : {cap_str}",
        f"  FP_MEMBRANE/PROC : {FP_MEMBRANE} / {FP_PROCESS} pN",
        f"  SMOOTH_SIGMA     : {SMOOTH_SIGMA}  "
        f"BOUNDARY_SMOOTH: {BOUNDARY_SMOOTH_SIGMA}",
        f"  RADIAL_FRAC_MIN  : {RADIAL_FRACTION_MIN}",
        f"  STEP             : {STEP} um  nbr={NEIGHBOUR_FRACTION}",
        f"  Max extension    : {max_ext:.3f} um",
        f"  Max n_procs      : {max_nproc}",
        "","="*66]
    txt = "\n".join(lines)
    print("\n"+txt)
    with open(f"{OUT_DIR}/summary_v4.txt","w") as fh: fh.write(txt)


# =============================================================
#  FIGURES
# =============================================================

def _style(ax,title,xl='',yl=''):
    ax.set_facecolor('#111128')
    ax.tick_params(colors='#aaa',labelsize=9)
    ax.xaxis.label.set_color('#ccc'); ax.yaxis.label.set_color('#ccc')
    ax.set_title(title,color='white',fontsize=10,pad=6)
    if xl: ax.set_xlabel(xl,color='#ccc')
    if yl: ax.set_ylabel(yl,color='#ccc')
    for sp in ax.spines.values(): sp.set_color('#333355')

def _draw_p1(ax,b_hist,n_done):
    ns   = min(N_SNAPSHOTS,len(b_hist))
    idxs = np.linspace(0,len(b_hist)-1,ns,dtype=int)
    for ii,bi in enumerate(idxs):
        v=b_hist[bi]; cv=np.vstack([v,v[0]])
        f=ii/max(ns-1,1)
        ax.plot(cv[:,0],cv[:,1],color=plt.cm.plasma(f),
                alpha=0.35+0.65*f,lw=1.4)
    ax.set_aspect('equal')
    lim=CELL_R+MAX_EXT_UM+1.5; ax.set_xlim(-lim,lim); ax.set_ylim(-lim,lim)
    sm=plt.cm.ScalarMappable(cmap='plasma',
        norm=plt.Normalize(0,n_done*DT_EPOCH/60))
    sm.set_array([])
    cb=plt.colorbar(sm,ax=ax,fraction=0.046,pad=0.04)
    cb.set_label('Time (min)',color='#ccc',fontsize=8)
    cb.ax.tick_params(colors='#aaa',labelsize=7)
    _style(ax,f'Boundary evolution ({ns} snapshots)','x (um)','y (um)')

def _draw_p2(ax,proc_records,times_min):
    cols=plt.cm.tab10(np.linspace(0,0.75,8))
    max_lab=max((p['label'] for pr in proc_records for p in pr),default=0)
    for lab in range(1,min(max_lab+1,9)):
        series=[next((p['max_ext_um'] for p in pr if p['label']==lab),0.0)
                for pr in proc_records]
        if max(series)>PROC_THRESH_UM:
            ax.plot(times_min[:len(series)],series,
                    lw=1.8,color=cols[lab-1],label=f'P{lab}')
    cap_s=f"cap={MAX_PROCESSES}" if ENABLE_PROCESS_CAP else "no cap"
    ax.axhline(5.0,color='#D4537E',ls=':',lw=1,label='5 um (paper)')
    ax.legend(fontsize=7,framealpha=0.25,labelcolor='white',ncol=2)
    _style(ax,f'Per-process extension [{cap_s}]','Time (min)','Extension (um)')

def _draw_p3(ax,perim_hist,area_hist,times_min):
    ax.plot(times_min,perim_hist,color='#4488FF',lw=2,label='Perimeter')
    ax2=ax.twinx()
    ax2.plot(times_min,area_hist,color='#FF8844',lw=2,label='Area (um²)')
    ax2.tick_params(colors='#FF8844',labelsize=8)
    ax2.set_ylabel('Area (um²)',color='#FF8844')
    ax.legend(fontsize=7,loc='upper left',framealpha=0.25,labelcolor='white')
    ax2.legend(fontsize=7,loc='lower right',framealpha=0.25,labelcolor='#FF8844')
    _style(ax,'Perimeter & Area','Time (min)','Perimeter (um)')

def _draw_p4(ax,end_hist,times_min):
    sampled=[e for e in end_hist if e is not None]
    if not sampled: _style(ax,'Plus/Minus ends (no samples)','Time (min)',''); return
    ep_idxs=np.array([e[0] for e in sampled])
    np_arr=np.array([e[1] for e in sampled],dtype=float)
    nm_arr=np.array([e[2] for e in sampled],dtype=float)
    total=np_arr+nm_arr
    t=ep_idxs*DT_EPOCH/60.0
    w=(t[1]-t[0])*0.35 if len(t)>1 else END_SAMPLE_EVERY*DT_EPOCH/60*0.35
    ax.bar(t-w/2,nm_arr,width=w,color='red',alpha=0.75,label='minus')
    ax.bar(t+w/2,np_arr,width=w,color='#4488FF',alpha=0.75,label='plus')
    ax2=ax.twinx()
    ax2.plot(t,total,color='#44CC44',lw=2,marker='o',ms=4)
    ax2.tick_params(colors='#44CC44',labelsize=8)
    ax2.set_ylabel('total process MTs',color='#44CC44')
    ax.legend(fontsize=7,loc='upper left',framealpha=0.25,labelcolor='white')
    _style(ax,f'Plus/Minus ends [every {END_SAMPLE_EVERY} ep]',
           'Time (min)','# MT ends')

def _draw_p5(ax,b_hist,proc_records,dyn_idx_hist):
    v=b_hist[-1]; n=len(v)
    cv=np.vstack([v,v[0]])
    ax.plot(cv[:,0],cv[:,1],color='#2255AA',lw=1.2,alpha=0.5,zorder=2)
    if dyn_idx_hist:
        for vi in dyn_idx_hist[-1]:
            ax.plot(v[vi][0],v[vi][1],'o',color='#4488FF',ms=5,
                    zorder=5,alpha=0.9)
    procs=proc_records[-1] if proc_records else []
    cmap=plt.cm.Set1
    for pi,p in enumerate(procs):
        idx=p['vertex_ids']
        color=cmap(pi/max(len(procs),1))
        lo=(idx[0]-1)%n; hi=(idx[-1]+1)%n
        pts=v[[lo]+idx+[hi]]
        from matplotlib.patches import Polygon as MplPoly
        ax.add_patch(MplPoly(np.vstack([pts,[0,0]]),closed=True,
                             facecolor=color,alpha=0.22,
                             edgecolor=color,linewidth=0,zorder=3))
        s2=np.vstack([pts,pts[0]])
        ax.plot(s2[:,0],s2[:,1],color=color,lw=2.5,zorder=4,
                label=f"P{p['label']} {p['max_ext_um']:.2f}µm")
        ax.plot(p['tip_x'],p['tip_y'],'*',color=color,ms=12,zorder=6)
    ax.set_aspect('equal')
    lim=CELL_R+MAX_EXT_UM+1.5; ax.set_xlim(-lim,lim); ax.set_ylim(-lim,lim)
    if procs: ax.legend(fontsize=7,framealpha=0.25,labelcolor='white',
                        loc='lower right')
    _style(ax,'Processes + dyneins (★=tip, ●=dynein)','x (um)','y (um)')

def _draw_p6(ax,f_hist,fe):
    na=len(f_hist)
    if na==0: ax.set_title('Polar confine-force (no data)',color='white'); return
    ns=max(1,min(FORCE_PLOT_N_EPOCHS,na))
    eps=(list(range(max(0,na-ns),na)) if ns>1 else [fe])
    ang=np.linspace(0,2*np.pi,N_SEGMENTS,endpoint=False)
    w=2*np.pi/N_SEGMENTS; ne=len(eps)
    for rank,ep_idx in enumerate(eps):
        if ep_idx<0 or ep_idx>=na: continue
        f=np.maximum(f_hist[ep_idx],0)
        if len(f)!=N_SEGMENTS:
            f=np.interp(ang,np.linspace(0,2*np.pi,len(f),endpoint=False),f)
        fmax=f.max()
        if fmax<=0: continue
        alpha=0.20+0.70*(rank/max(ne-1,1))
        ax.bar(ang,f,width=w,color=plt.cm.plasma(f/(fmax+1e-9)),
               alpha=alpha,zorder=rank+1)
        ax.text(ang[int(np.argmax(f))],fmax*1.05,f'ep{ep_idx+1}',
                color='white',fontsize=6,ha='center',va='bottom',
                alpha=min(alpha+0.2,1.0))
    ax.plot(np.linspace(0,2*np.pi,300),[FP_MEMBRANE]*300,'r--',lw=1,alpha=0.8)
    ax.tick_params(colors='#aaa',labelsize=7)
    er=(f"ep{eps[0]+1}–{eps[-1]+1}" if len(eps)>1 else f"ep{eps[0]+1}")
    ax.set_title(f'Polar confine-force ({er})',color='white',fontsize=9)

def plot_report(b_hist,f_hist,proc_records,dyn_idx_hist,
                perim_hist,area_hist,end_hist):
    os.makedirs(OUT_DIR,exist_ok=True)
    n_done=len(f_hist); fe=_fe(n_done)
    times_min=np.arange(len(b_hist))*DT_EPOCH/60
    cap_s=f"cap={MAX_PROCESSES}" if ENABLE_PROCESS_CAP else "no_cap"
    suptitle=(f'Oelz 2018 v4 | boost={PROCESS_TIP_BOOST}pN '
              f'proc_{cap_s} enrich={DYNEIN_PROCESS_ENRICHMENT}x '
              f'carry_dyn={CARRY_DYNEIN_POSITIONS} '
              f'carry_kin={CARRY_KINESIN_POSITIONS}')
    draws=[
        ('p1',lambda ax:_draw_p1(ax,b_hist,n_done),          False),
        ('p2',lambda ax:_draw_p2(ax,proc_records,times_min),  False),
        ('p3',lambda ax:_draw_p3(ax,perim_hist,area_hist,times_min),False),
        ('p4',lambda ax:_draw_p4(ax,end_hist,times_min),      False),
        ('p5',lambda ax:_draw_p5(ax,b_hist,proc_records,dyn_idx_hist),False),
        ('p6',lambda ax:_draw_p6(ax,f_hist,fe),               True),
    ]
    names=['boundary','process_ext','perimeter_area',
           'end_polarity','processes_dynein','polar_force']
    for (tag,fn,polar),name in zip(draws,names):
        fig_s=plt.figure(figsize=(6.5,5.5))
        fig_s.patch.set_facecolor('#0d0d1a')
        ax_s=fig_s.add_subplot(111,projection='polar' if polar else None)
        if not polar: ax_s.set_facecolor('#111128')
        fn(ax_s); fig_s.tight_layout()
        fig_s.savefig(f"{OUT_DIR}/panel_{tag}_{name}_v4.png",
                      dpi=150,bbox_inches='tight',facecolor='#0d0d1a')
        plt.close(fig_s)
        print(f"[OK] panel_{tag}_{name}_v4.png")
    fig=plt.figure(figsize=(17,11))
    fig.patch.set_facecolor('#0d0d1a')
    gs=gridspec.GridSpec(2,3,figure=fig,hspace=0.42,wspace=0.33)
    _draw_p1(fig.add_subplot(gs[0,0]),b_hist,n_done)
    _draw_p2(fig.add_subplot(gs[0,1]),proc_records,times_min)
    _draw_p3(fig.add_subplot(gs[0,2]),perim_hist,area_hist,times_min)
    _draw_p4(fig.add_subplot(gs[1,0]),end_hist,times_min)
    _draw_p5(fig.add_subplot(gs[1,1]),b_hist,proc_records,dyn_idx_hist)
    ax6=fig.add_subplot(gs[1,2],projection='polar')
    ax6.set_facecolor('#111128')
    _draw_p6(ax6,f_hist,fe)
    fig.suptitle(suptitle,color='white',fontsize=9,y=0.998)
    fig.savefig(f"{OUT_DIR}/protrusion_report_v4.png",
                dpi=150,bbox_inches='tight',facecolor='#0d0d1a')
    plt.close(fig)
    print(f"[OK] protrusion_report_v4.png")


# =============================================================
#  MAIN
# =============================================================

def main():
    os.makedirs(EPOCH_DIR,exist_ok=True); os.makedirs(OUT_DIR,exist_ok=True)
    cap_s=f"ON MAX={MAX_PROCESSES}" if ENABLE_PROCESS_CAP else "OFF"
    print("="*66)
    print("  Oelz 2018 — Protrusion v4")
    print(f"  {N_EPOCHS} epochs × {DT_EPOCH}s  dt={_dt():.4f}s")
    print(f"  carry: dynein={CARRY_DYNEIN_POSITIONS}  "
          f"kinesin={CARRY_KINESIN_POSITIONS}")
    print(f"  dynein enrichment={DYNEIN_PROCESS_ENRICHMENT}x  "
          f"tip_boost={PROCESS_TIP_BOOST}pN  process_cap={cap_s}")
    print("="*66)

    if not os.path.isfile(TEMPLATE):
        raise FileNotFoundError(f"Template not found: {TEMPLATE}")

    cytosim_ok=(os.path.isfile(CYTOSIM_BIN) and os.access(CYTOSIM_BIN,os.X_OK))
    report_ok =(os.path.isfile(REPORT_BIN)  and os.access(REPORT_BIN, os.X_OK))
    if not cytosim_ok: print(f"\n[!] Cytosim not found — SYNTHETIC mode\n")
    if cytosim_ok and not report_ok: print(f"\n[!] Report binary missing\n")

    verts              = make_circle()
    dynein_indices     = _place_dyneins_initial(verts)
    prev_fiber_points  = None
    prev_dynein_xy     = None   # (x,y) list from dynein_cortex:position
    prev_kinesin_xy    = None   # (x,y) list from kinesin1:state

    print(f"  [dynein init] {len(dynein_indices)} dyneins placed")

    b_hist       = [verts.copy()]
    f_hist       = []
    proc_records = [[]]
    dyn_idx_hist = [list(dynein_indices)]
    perim_hist   = [polygon_perimeter(verts)]
    area_hist    = [polygon_area(verts)]
    end_hist     = []

    for ep in range(N_EPOCHS):
        perim      = polygon_perimeter(verts)
        procs_prev = proc_records[-1]
        cap_active = (ENABLE_PROCESS_CAP and MAX_PROCESSES>0
                      and len(procs_prev)>=MAX_PROCESSES)
        print(f"\n{'='*66}")
        print(f"  EPOCH {ep+1:3d}/{N_EPOCHS}  "
              f"t={ep*DT_EPOCH:.0f}-{(ep+1)*DT_EPOCH:.0f}s  "
              f"N_dyn={len(dynein_indices)}  perim={perim:.1f}um  "
              f"n_procs={len(procs_prev)}  "
              f"cap={'ACTIVE' if cap_active else 'off'}")
        print(f"{'='*66}")

        if cytosim_ok:
            cym_path,ep_dir = write_cym(
                ep, verts, dynein_indices,
                prev_fiber_points=prev_fiber_points,
                prev_dynein_xy   =prev_dynein_xy,
                prev_kinesin_xy  =prev_kinesin_xy,
            )
            ok = run_cytosim(cym_path, ep_dir)

            if ok and report_ok:
                paths             = run_report(ep_dir, ep)
                forces_df         = load_forces(ep_dir, ep)
                prev_fiber_points = parse_fiber_points(paths['points'])
                ends_recs         = parse_fiber_ends(paths['ends'])
                # Carry motor positions
                prev_dynein_xy  = (parse_dynein_positions(paths['dynein'])
                                   if ENABLE_DYNEIN else None)
                prev_kinesin_xy = (parse_kinesin_positions(paths['kinesin'])
                                   if ENABLE_KINESIN and CARRY_KINESIN_POSITIONS
                                   else None)
                print(f"    [carry] dynein={len(prev_dynein_xy or [])}  "
                      f"kinesin={len(prev_kinesin_xy or [])}")
            elif ok:
                print("  [!] report binary missing")
                forces_df=pd.DataFrame(); prev_fiber_points=None
                ends_recs=[]; prev_dynein_xy=None; prev_kinesin_xy=None
            else:
                print("  [!] Cytosim failed — synthetic fallback")
                forces_df=synthetic_forces(verts,ep); prev_fiber_points=None
                ends_recs=[]; prev_dynein_xy=None; prev_kinesin_xy=None
        else:
            forces_df=synthetic_forces(verts,ep); prev_fiber_points=None
            ends_recs=[]; prev_dynein_xy=None; prev_kinesin_xy=None
            ep_dir=os.path.join(EPOCH_DIR,f"epoch_{ep:04d}")

        verts,seg_f,thresh,n_act = apply_protrusion(verts,forces_df,procs_prev)

        # Update dynein indices (process-biased, position-enriched)
        dynein_indices = _update_dynein_positions(verts,dynein_indices,procs_prev)

        procs = detect_processes(verts)
        mx    = max((p['max_ext_um'] for p in procs),default=0.0)

        n_plus,n_minus = (classify_ends_at_tips(ends_recs,procs)
                          if ep%END_SAMPLE_EVERY==0 else (None,None))

        r_d   = np.linalg.norm(verts,axis=1)
        ext_d = np.maximum(0.0,r_d-CELL_R)
        ip    = ext_d>PROC_THRESH_UM; ib=~ip
        th_b  = thresh[ib]; th_p = thresh[ip]
        ten   = (float(thresh[ib].min())-FP_MEMBRANE if ib.any() else
                 float(thresh[ip].min())-FP_PROCESS   if ip.any() else 0.0)

        print(f"  active={n_act}  max_F={seg_f.max():.2f}pN  "
              f"max_ext={mx:.3f}um  n_procs={len(procs)}")
        if n_plus is not None:
            print(f"  ends: +={n_plus}  -={n_minus}")
        print(f"  tension_penalty=+{ten:.3f}pN")
        if th_b.size>0:
            print(f"  MEMBRANE({ib.sum():3d}): "
                  f"min={th_b.min():.2f} mean={th_b.mean():.2f} pN")
        if th_p.size>0:
            print(f"  PROCESS ({ip.sum():3d}): "
                  f"min={th_p.min():.2f} mean={th_p.mean():.2f} pN")
        print_processes(procs)

        b_hist.append(verts.copy())
        f_hist.append(seg_f.copy())
        proc_records.append(procs)
        dyn_idx_hist.append(list(dynein_indices))
        perim_hist.append(polygon_perimeter(verts))
        area_hist.append(polygon_area(verts))
        end_hist.append((ep,n_plus,n_minus,n_plus+n_minus)
                        if n_plus is not None else None)

    print(f"\n{'='*66}")
    print("  [STEP 7] Saving ...")
    save_outputs(b_hist,f_hist,proc_records,dyn_idx_hist,
                 perim_hist,area_hist,end_hist)
    print("  [STEP 7] Plotting ...")
    plot_report(b_hist,f_hist,proc_records,dyn_idx_hist,
                perim_hist,area_hist,end_hist)
    print("[OK] Done.")


if __name__ == "__main__":
    main()