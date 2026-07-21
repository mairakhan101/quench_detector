'''
What you get:
1) Disturbance (dipole-like) field map: |E| magnitude + streamlines + arrows using two infinite current lines (close together)
2) Overlay of each QA winding’s vertices (placed coil) on the field map
3) “What QA measures at each point”:
   - Avp(z_k) per vertex (complex)
   - signed contribution Avp(z_k)*sign_k per vertex (complex)
   - plots vs theta (sorted for visualization) AND colored points on the coil
4) Winding output per file: total complex response, magnitude & phase
5) C6p, C8p and inferred location qloc, plus qloc vs sources plot
'''

import os
import numpy as np
import matplotlib.pyplot as plt


# ----------------------------
# Configuration
# ----------------------------
DATA_DIR_CANDIDATES = [".", "/mnt/data"]

FILES = {
    "6p_norm": "coilparam_MQXFA_test_6p_norm.dat",
    "6p_skew": "coilparam_MQXFA_test_6p_skew.dat",
    "8p_norm": "coilparam_MQXFA_test_8p_norm.dat",
    "8p_skew": "coilparam_MQXFA_test_8p_skew.dat",
}

# Disturbance sources (MATLAB pointsZi)
pointsZi = np.array([0.0 + 0.075j, 0.0 + 0.076j], dtype=np.complex128)

# “Measurement radius” scaling used in MATLAB
measRadius = 0.048383

# Orientation angle (MATLAB uses 0)
orientAngle = 0.0  # radians

# QA placement (YOU CAN CHANGE THESE)
# z0 = x0 + i*y0 in meters (placement of antenna center)
QA_Z0 = 0.0 + 0.0j
QA_ROT = 0.0  # radians

# Field-map view limits (tune if you like)
FIELD_XLIM = (-0.08, 0.08)
FIELD_YLIM = (-0.02, 0.12)



def find_data_dir() -> str:
    for d in DATA_DIR_CANDIDATES:
        ok = True
        for fn in FILES.values():
            if not os.path.exists(os.path.join(d, fn)):
                ok = False
                break
        if ok:
            return d
    raise FileNotFoundError(
        "Could not find all coilparam files. Searched:\n" + "\n".join(DATA_DIR_CANDIDATES)
    )


def read_coilparam(path: str):
    """
    Parses lines like:
      length    nturns    R      Theta
      VERTEX RBUCK  1.0000000000e+00  1.00  4.8500000000e-02  7.4176493210e-01

    Returns arrays:
      lengths, nturns, R, theta, header
    """
    lengths, nturns, R, theta = [], [], [], []
    header = None
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            if header is None:
                header = line
                continue
            parts = line.split()
            if len(parts) < 6:
                continue
            if parts[0].upper() != "VERTEX":
                continue
            try:
                L = float(parts[2])
                N = float(parts[3])
                r = float(parts[4])
                th = float(parts[5])
            except ValueError:
                continue
            lengths.append(L)
            nturns.append(N)
            R.append(r)
            theta.append(th)

    if len(R) == 0:
        raise ValueError(f"No coil vertices parsed from file: {path}")

    return (
        np.array(lengths, dtype=float),
        np.array(nturns, dtype=float),
        np.array(R, dtype=float),
        np.array(theta, dtype=float),
        header,
    )


def A_complex_potential(Z: np.ndarray, pointsZi_: np.ndarray) -> np.ndarray:
    """
    Same as MATLAB fluxCalc inner sum, but vectorized:
      Avp(z) = Σ_j [ ((-1)^j)*(-1) * log(z - p_j) ]  with MATLAB j starting at 1
    """
    Z = np.asarray(Z, dtype=np.complex128)
    pointsZi_ = np.asarray(pointsZi_, dtype=np.complex128).reshape(-1)
    A = np.zeros_like(Z, dtype=np.complex128)
    for j, p in enumerate(pointsZi_, start=1):
        A += ((-1) ** j) * (-1) * np.log(Z - p)
    return A


def dAdz(Z: np.ndarray, pointsZi_: np.ndarray) -> np.ndarray:
    """
    Derivative of complex potential:
      dA/dz = Σ_j [ ((-1)^j)*(-1) * 1/(z - p_j) ]
    """
    Z = np.asarray(Z, dtype=np.complex128)
    pointsZi_ = np.asarray(pointsZi_, dtype=np.complex128).reshape(-1)
    out = np.zeros_like(Z, dtype=np.complex128)
    for j, p in enumerate(pointsZi_, start=1):
        out += ((-1) ** j) * (-1) * (1.0 / (Z - p))
    return out


def alternating_signs(n: int) -> np.ndarray:
    """MATLAB alt_pm1: [+1, -1, +1, -1, ...] in FILE ORDER."""
    s = np.ones(n, dtype=float)
    s[1::2] *= -1.0
    return s


def place_vertices(R: np.ndarray, theta: np.ndarray, *, z0: complex, rotation: float) -> np.ndarray:
    """
    Place coil geometry in space:
      z_placed = z0 + exp(i*rotation) * (R * exp(i*theta))
    """
    z_local = R * np.exp(1j * theta)
    return z0 + np.exp(1j * rotation) * z_local


def qa_response_from_disturbance(R, theta, pointsZi_, *, z0=0.0+0.0j, rotation=0.0):
    """
    Returns:
      z_placed: complex vertex positions after placement
      Avp: A(z) per vertex (complex)
      signs: +/- pattern (file order)
      contrib: Avp*signs (complex)
      total: sum(contrib) (complex)   <-- this is the winding output
    """
    z_placed = place_vertices(R, theta, z0=z0, rotation=rotation)
    Avp = A_complex_potential(z_placed, pointsZi_)
    signs = alternating_signs(len(Avp))
    contrib = Avp * signs
    total = np.sum(contrib)
    return z_placed, Avp, signs, contrib, total
def winding_total_for_file(data_dir, key, z0, rotation):
    # read coil file
    _, _, R, theta, _ = read_coilparam(os.path.join(data_dir, FILES[key]))
    theta = theta - orientAngle
    _, _, _, _, total = qa_response_from_disturbance(
    R, theta, pointsZi, z0=z0, rotation=rotation
    )

    return total


def compute_all_totals_and_qloc(data_dir, z0, rotation):
    tot_6n = winding_total_for_file(data_dir, "6p_norm", z0, rotation)
    tot_6s = winding_total_for_file(data_dir, "6p_skew", z0, rotation)
    tot_8n = winding_total_for_file(data_dir, "8p_norm", z0, rotation)
    tot_8s = winding_total_for_file(data_dir, "8p_skew", z0, rotation)

    C6p = np.real(tot_6n) + 1j*np.real(tot_6s)
    C8p = np.real(tot_8n) + 1j*np.real(tot_8s)
    qloc = (4.0/3.0) * (C6p / C8p) * measRadius

    return {
        "tot_6n": tot_6n, "tot_6s": tot_6s, "tot_8n": tot_8n, "tot_8s": tot_8s,
        "C6p": C6p, "C8p": C8p, "qloc": qloc
    }


# Plotting
def plot_field_vector_map(pointsZi_, *, xlim, ylim, n=320, arrow_step=12, title="Disturbance field"):
    x = np.linspace(xlim[0], xlim[1], n)
    y = np.linspace(ylim[0], ylim[1], n)
    X, Y = np.meshgrid(x, y)
    Z = X + 1j * Y

    deriv = dAdz(Z, pointsZi_)
    Ex = -np.real(deriv)
    Ey =  np.imag(deriv)
    Emag = np.sqrt(Ex**2 + Ey**2)

    # Mask close to singularities
    mask = np.zeros_like(Z, dtype=bool)
    for p in pointsZi_:
        mask |= np.abs(Z - p) < 5e-4
    Exm = np.ma.array(Ex, mask=mask)
    Eym = np.ma.array(Ey, mask=mask)
    Em  = np.ma.array(Emag, mask=mask)

    # Magnitude + streamlines
    plt.figure(figsize=(8, 6))
    plt.contourf(X, Y, Em, levels=70)
    plt.colorbar(label="|E| (field magnitude, arbitrary units)")
    plt.streamplot(X, Y, Exm.filled(0.0), Eym.filled(0.0), density=1.3)
    plt.scatter(np.real(pointsZi_), np.imag(pointsZi_), marker="s", s=90, label="disturbance sources")
    plt.title(title)
    plt.xlabel("x (m)")
    plt.ylabel("y (m)")
    plt.axis("equal")
    plt.legend()
    plt.show()

    # Magnitude + arrows
    plt.figure(figsize=(8, 6))
    plt.contourf(X, Y, Em, levels=70)
    plt.colorbar(label="|E| (field magnitude, arbitrary units)")
    xs = X[::arrow_step, ::arrow_step]
    ys = Y[::arrow_step, ::arrow_step]
    exs = Exm[::arrow_step, ::arrow_step].filled(0.0)
    eys = Eym[::arrow_step, ::arrow_step].filled(0.0)
    plt.quiver(xs, ys, exs, eys, angles="xy")
    plt.scatter(np.real(pointsZi_), np.imag(pointsZi_), marker="s", s=90, label="disturbance sources")
    plt.title(title + " (arrows)")
    plt.xlabel("x (m)")
    plt.ylabel("y (m)")
    plt.axis("equal")
    plt.legend()
    plt.show()


def overlay_coil_on_field(pointsZi_, z_coil, *, xlim, ylim, n=320, title="Coil sampling the disturbance"):
    """Overlay the placed coil vertices on a |E| magnitude map."""
    x = np.linspace(xlim[0], xlim[1], n)
    y = np.linspace(ylim[0], ylim[1], n)
    X, Y = np.meshgrid(x, y)
    Z = X + 1j * Y

    deriv = dAdz(Z, pointsZi_)
    Ex = -np.real(deriv)
    Ey =  np.imag(deriv)
    Emag = np.sqrt(Ex**2 + Ey**2)

    mask = np.zeros_like(Z, dtype=bool)
    for p in pointsZi_:
        mask |= np.abs(Z - p) < 5e-4
    Em = np.ma.array(Emag, mask=mask)

    plt.figure(figsize=(8, 6))
    plt.contourf(X, Y, Em, levels=70)
    plt.colorbar(label="|E| (arbitrary units)")
    plt.plot(np.real(z_coil), np.imag(z_coil), "o", label="QA vertices (placed)")
    plt.scatter(np.real(pointsZi_), np.imag(pointsZi_), marker="s", s=90, label="disturbance sources")
    plt.title(title)
    plt.xlabel("x (m)")
    plt.ylabel("y (m)")
    plt.axis("equal")
    plt.legend()
    plt.show()


def plot_coil_geometry_pairs(R, theta, title):
    x = R * np.cos(theta)
    y = R * np.sin(theta)
    plt.figure()
    for i in range(0, len(R) - 1, 2):
        plt.plot(x[i:i+2], y[i:i+2], "-o")
    plt.plot(x[0::2], y[0::2], "rs", label="odd index (MATLAB 1,3,5...)")
    plt.plot(x[1::2], y[1::2], "ks", label="even index (MATLAB 2,4,6...)")
    plt.title(title.replace("_", r"\_"))
    plt.axis("equal")
    plt.xlabel("x (local)")
    plt.ylabel("y (local)")
    plt.legend()
    plt.show()


def plot_vs_theta_sorted(theta, y, title, ylabel):
    idx = np.argsort(theta)
    plt.figure()
    plt.plot(theta[idx], y[idx], "-o")
    plt.title(title + " (theta-sorted for visualization)")
    plt.xlabel("theta (rad)")
    plt.ylabel(ylabel)
    plt.show()


def plot_complex_point(z, title):
    plt.figure()
    plt.scatter([np.real(z)], [np.imag(z)], s=140)
    plt.axhline(0); plt.axvline(0)
    plt.axis("equal")
    plt.title(title)
    plt.xlabel("Re")
    plt.ylabel("Im")
    plt.show()


def plot_colored_vertices(z_coil, values, title, cbar_label):
    """
    Shows “each point in the QA measured and the response”:
    coil vertices colored by |value| or Re(value).
    """
    plt.figure(figsize=(7, 6))
    sc = plt.scatter(np.real(z_coil), np.imag(z_coil), c=values, s=45)
    plt.colorbar(sc, label=cbar_label)
    plt.title(title)
    plt.xlabel("x (m)")
    plt.ylabel("y (m)")
    plt.axis("equal")
    plt.show()

def run_winding(file_path, label, *, z0, rotation):
    L, N, R, theta, header = read_coilparam(file_path)

    # MATLAB: QA.data(:,4) = QA.data(:,4) - orientAngle
    theta = theta - orientAngle

    # Geometry pairing plot in local coordinates (file order pairing)
    plot_coil_geometry_pairs(R, theta, f"{label}: coil geometry (paired segments, local coords)")

    # Compute QA response at placed coil vertices
    z_placed, Avp, signs, contrib, total = qa_response_from_disturbance(
        R, theta, pointsZi, z0=z0, rotation=rotation
    )

    # Overlay coil on field magnitude map (so you SEE where it samples)
    overlay_coil_on_field(
        pointsZi, z_placed, xlim=FIELD_XLIM, ylim=FIELD_YLIM,
        title=f"{label}: placed QA vertices over |E| map"
    )

    # What each point “measures” (per vertex):
    # 1) Raw Avp along theta
    plot_vs_theta_sorted(theta, np.real(Avp), f"{label}: Re(Avp) per vertex", "Re(Avp)")
    plot_vs_theta_sorted(theta, np.imag(Avp), f"{label}: Im(Avp) per vertex", "Im(Avp)")

    # 2) Signed contribution Avp*signs (this is what gets summed)
    plot_vs_theta_sorted(theta, np.real(contrib), f"{label}: Re(Avp*sign) (QA contribution)", "Re(contrib)")
    plot_vs_theta_sorted(theta, np.imag(contrib), f"{label}: Im(Avp*sign) (QA contribution)", "Im(contrib)")

    # 3) Visual picture on the coil itself:
    plot_colored_vertices(
        z_placed, np.abs(Avp),
        title=f"{label}: coil vertices colored by |Avp(z)| (raw sample strength)",
        cbar_label="|Avp| (arb)"
    )
    plot_colored_vertices(
        z_placed, np.abs(contrib),
        title=f"{label}: coil vertices colored by |Avp*sign| (contribution strength)",
        cbar_label="|contrib| (arb)"
    )

    # Winding output (complex “channel”)
    mag = np.abs(total)
    ph = np.angle(total)
    print(f"\n[{label}] total = {total}")
    print(f"[{label}] |total| = {mag}  phase(rad) = {ph}")

    plot_complex_point(total, f"{label}: winding output (complex)")

    # Save MATLAB-like flux_ file (theta, real(contrib)) in FILE ORDER
    out = np.column_stack([theta, np.real(contrib)])
    out_name = "flux_" + os.path.basename(file_path)
    np.savetxt(out_name, out, delimiter=" ", fmt="%.10e")
    print(f"[{label}] wrote {out_name}")

    return total



def main():
    data_dir = find_data_dir()
    print("Using data directory:", data_dir)

    # 1) Disturbance field visualization (truth in space)
    plot_field_vector_map(
        pointsZi,
        xlim=FIELD_XLIM, ylim=FIELD_YLIM,
        title="Disturbance field derived from complex potential (|E| + streamlines)"
    )

    # 2) Run all four QA windings at the chosen placement
    tot_6n = run_winding(os.path.join(data_dir, FILES["6p_norm"]), "6p normal", z0=QA_Z0, rotation=QA_ROT)
    tot_6s = run_winding(os.path.join(data_dir, FILES["6p_skew"]), "6p skew",   z0=QA_Z0, rotation=QA_ROT)
    tot_8n = run_winding(os.path.join(data_dir, FILES["8p_norm"]), "8p normal", z0=QA_Z0, rotation=QA_ROT)
    tot_8s = run_winding(os.path.join(data_dir, FILES["8p_skew"]), "8p skew",   z0=QA_Z0, rotation=QA_ROT)

    # 3) MATLAB-style combination into complex harmonics
    C6p = np.real(tot_6n) + 1j * np.real(tot_6s)
    C8p = np.real(tot_8n) + 1j * np.real(tot_8s)
    qloc = (4.0 / 3.0) * (C6p / C8p) * measRadius

    print("\n--- Harmonics / Localization (MATLAB-equivalent) ---")
    print("C6p  =", C6p, " |C6p|=", np.abs(C6p), " phase=", np.angle(C6p))
    print("C8p  =", C8p, " |C8p|=", np.abs(C8p), " phase=", np.angle(C8p))
    print("qloc =", qloc)

    # 4) Summary plots
    plt.figure()
    plt.plot([0, 1], [np.abs(C6p), np.abs(C8p)], "-o")
    plt.xticks([0, 1], ["|C6p|", "|C8p|"])
    plt.ylabel("magnitude (arb)")
    plt.title("Harmonic magnitudes")
    plt.show()

    plt.figure()
    plt.plot([0, 1], [np.angle(C6p), np.angle(C8p)], "-o")
    plt.xticks([0, 1], ["∠C6p", "∠C8p"])
    plt.ylabel("phase (rad)")
    plt.title("Harmonic phases")
    plt.show()

    # 5) qloc vs sources in the complex plane
    plt.figure(figsize=(7, 6))
    plt.scatter(np.real(qloc), np.imag(qloc), s=140, marker="o", label="Estimated qloc")
    plt.scatter(np.real(pointsZi), np.imag(pointsZi), s=90, marker="s", label="True disturbance sources")
    plt.axhline(0, linewidth=0.8)
    plt.axvline(0, linewidth=0.8)
    plt.axis("equal")
    plt.xlabel("x (m)")
    plt.ylabel("y (m)")
    plt.title("Estimated disturbance location (qloc) vs true sources")
    plt.legend()
    plt.grid(True)
    plt.show()

    rotation = QA_ROT  # keep fixed for now
    
    # Sweep +/- 5 mm in x and y (edit as needed)
    xs = np.linspace(-0.005, 0.005, 21)
    ys = np.linspace(-0.005, 0.005, 21)
    
    mag6 = np.zeros((len(ys), len(xs)))
    mag8 = np.zeros((len(ys), len(xs)))
    qloc_x = np.zeros((len(ys), len(xs)))
    qloc_y = np.zeros((len(ys), len(xs)))
    
    for iy, y in enumerate(ys):
        for ix, x in enumerate(xs):
            z0 = x + 1j*y
            out = compute_all_totals_and_qloc(data_dir, z0=z0, rotation=rotation)
            mag6[iy, ix] = np.abs(out["C6p"])
            mag8[iy, ix] = np.abs(out["C8p"])
            qloc_x[iy, ix] = np.real(out["qloc"])
            qloc_y[iy, ix] = np.imag(out["qloc"])
    
    # Heatmap: |C6p|
    plt.figure(figsize=(7, 5))
    plt.imshow(mag6, origin="lower",
               extent=[xs[0], xs[-1], ys[0], ys[-1]], aspect="auto")
    plt.colorbar(label="|C6p| (arb)")
    plt.title("Sensitivity map: |C6p| vs QA placement (z0)")
    plt.xlabel("x0 (m)")
    plt.ylabel("y0 (m)")
    plt.show()
    
    # Heatmap: |C8p|
    plt.figure(figsize=(7, 5))
    plt.imshow(mag8, origin="lower",
               extent=[xs[0], xs[-1], ys[0], ys[-1]], aspect="auto")
    plt.colorbar(label="|C8p| (arb)")
    plt.title("Sensitivity map: |C8p| vs QA placement (z0)")
    plt.xlabel("x0 (m)")
    plt.ylabel("y0 (m)")
    plt.show()
    
    # Quiver: how qloc changes (vector field)
    plt.figure(figsize=(7, 6))
    Xg, Yg = np.meshgrid(xs, ys)
    plt.quiver(Xg, Yg, qloc_x, qloc_y, angles="xy")
    plt.title("Inferred disturbance location qloc vs QA placement")
    plt.xlabel("x0 (m)")
    plt.ylabel("y0 (m)")
    plt.axis("equal")
    plt.show()


    return qloc


if __name__ == "__main__":
    main()
