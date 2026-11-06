import tkinter as tk
from tkinter import ttk
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.patches import Rectangle, Circle

# =========================================================================
# KONFIGURACE PRO MENTORA & BAREVNÉ SCHÉMA (DARK UI + MCLAREN PAPAYA ORANGE)
# =========================================================================
ADMIN_AUTO_OPTIMIZE = True  # Přepni na True pro zobrazení admin tlačítka auto-optimalizace
DEBUG_PRINTS = True

# McLaren Papaya Dark Theme Palette
BG_DARK = "#121316"       # Hlavní tmavé pozadí
BG_PANEL = "#1B1D24"      # Pozadí postranních panelů a grafů
BG_INPUT = "#262933"      # Pozadí vstupních polí
FG_TEXT = "#F5F6FA"       # Hlavní bílý text
FG_MUTED = "#A4B0BE"      # Tlumený šedý popisný text
PAPAYA = "#FF8000"        # McLaren Papaya Orange (hlavní akcent)
PAPAYA_LIGHT = "#FFA54C"  # Světlejší oranžová pro hover/fill
CYAN_ACCENT = "#00D2D3"   # Kontrastní tyrkysová pro sekundární křivky
GREEN_OK = "#2ECC71"      # Zelená (tah / bezpečný stav)
RED_WARN = "#FF3838"      # Červená (suck-back / průraz MOSFETu / pád)
GRID_COL = "#2F3542"      # Barva mřížky v grafech

# Kompatibilita integrace pro NumPy 1.x i 2.x
trapz_func = getattr(np, "trapezoid", getattr(np, "trapz", None))

def dprint(msg):
    if DEBUG_PRINTS:
        print(f"[DEBUG] {msg}")

# =========================================================================
# PŘESNÝ 3D BIOT-SAVARTŮV INTEGRÁL PRO KRUHOVÉ SMYČKY V ROVINĚ Y-Z (OSA X)
# =========================================================================
def _pole_kruhove_smycky_3d(X_m, Z_m, x_loop_m, r_loop_m, I_amp):
    """
    Spočítá přesné složky magnetické indukce Bx, Bz v rovině Y=0 od kruhové smyčky
    o poloměru r_loop_m se středem v (x_loop_m, 0, 0), jejíž osa míří podél osy X.
    Integrace probíhá přes 24 úhlových segmentů phi kolem obvodu smyčky.
    """
    mu_0_4pi = 1e-7  # mu_0 / (4 * pi)
    n_phi = 24
    dphi = 2.0 * np.pi / n_phi
    phis = np.linspace(0.5 * dphi, 2.0 * np.pi - 0.5 * dphi, n_phi)

    dx = X_m - x_loop_m
    Bx_sum = np.zeros_like(X_m, dtype=float)
    Bz_sum = np.zeros_like(Z_m, dtype=float)

    for phi in phis:
        sin_p = np.sin(phi)
        cos_p = np.cos(phi)
        # Vektor R od elementu drátu (x_loop, r*cos(phi), r*sin(phi)) k bodu (X, 0, Z)
        rz = Z_m - r_loop_m * sin_p
        r2 = dx * dx + (r_loop_m * cos_p) ** 2 + rz * rz + 1e-9
        r3 = r2 * np.sqrt(r2)

        # Biot-Savartův vektorový součin (dl x R) v rovině Y=0:
        # dl = (0, -r*sin(phi)*dphi, r*cos(phi)*dphi)
        # (dl x R)_x = r * (r - Z*sin(phi)) * dphi
        # (dl x R)_z = r * dx * sin(phi) * dphi
        Bx_sum += (r_loop_m * (r_loop_m - Z_m * sin_p) * dphi) / r3
        Bz_sum += (r_loop_m * dx * sin_p * dphi) / r3

    faktor = mu_0_4pi * I_amp
    return faktor * Bx_sum, faktor * Bz_sum

# =========================================================================
# 1. FYZIKÁLNÍ JÁDRO - ZÁLOŽKA 1: ELEKTROMAGNETISMUS & XY-MOS BEZPEČNOST
# =========================================================================
def spocitej_fazi(params):
    dprint(
        f"Tab1: Přepočet fáze | V={params['napeti']}V, I_lim={params['proud_limit']}A, "
        f"Zapojení={params['zapojeni']}, R_stupne={params['r_stupne']}Ω, Z_off={params['z_vypnuti_mm']}mm"
    )
    mu_0 = 4.0 * np.pi * 1e-7

    r_in_m = (params["d_vnitrni"] / 2.0) / 1000.0
    r_out_m = (params["d_vnejsi"] / 2.0) / 1000.0
    sirka_m = params["sirka"] / 1000.0
    delka_dratu_1_m = params["delka_dratu"]

    # Počet závitů 1 cívky ze středního obvodu a délky drátu
    r_mid_m = (r_in_m + r_out_m) / 2.0
    obvod_stredni_m = 2.0 * np.pi * r_mid_m
    zavity_1_civka = max(5, int(round(delka_dratu_1_m / max(1e-4, obvod_stredni_m))))

    # Odpor celého stupně a proud s respektováním proudového omezovače (Current Limiter)
    r_stupne = max(0.02, params["r_stupne"])
    napeti = params["napeti"]
    proud_ohm = napeti / r_stupne
    proud_limit = max(0.5, params["proud_limit"])
    proud_celkem = min(proud_ohm, proud_limit)
    proud_omezen = proud_ohm > proud_limit

    # Rozdělení proudu do 1 cívky podle zapojení (Sériově vs. Paralelně)
    proud_1_civka = proud_celkem if params["zapojeni"] == "Sériově" else proud_celkem / 2.0

    # Indukčnost 1 cívky (Wheelerův vzorec v uH) a celého stupně
    r_cm = r_mid_m * 100.0
    len_cm = params["sirka"] / 10.0
    thick_cm = ((params["d_vnejsi"] - params["d_vnitrni"]) / 2.0) / 10.0
    jmenovatel = 9.0 * r_cm + 10.0 * len_cm + 11.0 * max(0.05, thick_cm)
    L_1_uH = (0.394 * (r_cm ** 2) * (zavity_1_civka ** 2)) / jmenovatel
    L_faze_uH = 2.0 * L_1_uH if params["zapojeni"] == "Sériově" else 0.5 * L_1_uH
    L_faze_H = L_faze_uH * 1e-6

    # Časové konstanty RL (sepnutí a vypnutí přes 1N5822 + 1.0 Ohm 3W rezistor)
    r_dump = 1.0
    v_schottky = 0.6
    tau_on_ms = (L_faze_H / r_stupne) * 1000.0
    tau_off_ms = (L_faze_H / (r_stupne + r_dump)) * 1000.0

    # Flyback špička na MOSFETu (XY-MOS AOD4184A má průrazné napětí V_DS_max = 40.0 V)
    v_flyback_spike = napeti + v_schottky + (proud_celkem * r_dump)
    mosfet_safe = v_flyback_spike <= 40.0

    # 2D Magnetické pole v řezu korytem (X = šířka koryta, Z = podél dráhy letu)
    xs_mm = np.linspace(-25, 25, 35)
    zs_mm = np.linspace(-25, 25, 35)
    X_mm, Z_mm = np.meshgrid(xs_mm, zs_mm)
    X_m, Z_m = X_mm / 1000.0, Z_mm / 1000.0

    mezera_mm = params["mezera"]
    x_leve_celo_mm = -mezera_mm / 2.0
    x_prave_celo_mm = mezera_mm / 2.0

    Bx = np.zeros_like(X_m)
    Bz = np.zeros_like(Z_m)

    # Diskretizace vinutí každé cívky na 3x2 prstenců a přesný 3D Biot-Savartův výpočet
    n_r, n_x = 3, 2
    I_ring = (zavity_1_civka * proud_1_civka) / (n_r * n_x)
    r_vals = np.linspace(r_in_m + 0.0005, max(r_in_m + 0.001, r_out_m - 0.0005), n_r)
    dx_vals = np.linspace(0.2 * sirka_m, 0.8 * sirka_m, n_x)

    for r_loop in r_vals:
        for dx_pos in dx_vals:
            for x_coil_pos in [x_leve_celo_mm / 1000.0 - dx_pos, x_prave_celo_mm / 1000.0 + dx_pos]:
                bx_i, bz_i = _pole_kruhove_smycky_3d(X_m, Z_m, x_coil_pos, r_loop, I_ring)
                Bx += bx_i
                Bz += bz_i

    B_intenzita = np.sqrt(Bx**2 + Bz**2)

    # Podélný profil Bx(z) v ose koryta (X = 0)
    zs_osa_mm = np.linspace(-40, 40, 321)
    zs_osa_m = zs_osa_mm / 1000.0
    X_zero = np.zeros_like(zs_osa_m)
    Bx_osa = np.zeros_like(zs_osa_m)

    for r_loop in r_vals:
        for dx_pos in dx_vals:
            for x_coil_pos in [x_leve_celo_mm / 1000.0 - dx_pos, x_prave_celo_mm / 1000.0 + dx_pos]:
                bx_i, _ = _pole_kruhove_smycky_3d(X_zero, zs_osa_m, x_coil_pos, r_loop, I_ring)
                Bx_osa += bx_i

    # Hranolový magnet (X = šířka v korytě, Y = výška, Z = délka ve směru jízdy)
    grade_map = {"N35": 1.19, "N38": 1.24, "N42": 1.30, "N52": 1.44}
    br = grade_map.get(params["mag_grade"], 1.24)

    mag_x_m = params["mag_x"] / 1000.0
    mag_y_m = params["mag_y"] / 1000.0
    mag_z_m = params["mag_z"] / 1000.0
    objem_mag_m3 = mag_x_m * mag_y_m * mag_z_m

    # Efektivní magnetizace otevřeného hranolu (započtení demagnetizačního faktoru tvaru N_d ≈ 1/3)
    k_tvaru = 0.65
    m_dipol = k_tvaru * (br / mu_0) * objem_mag_m3

    # Konečná délka magnetu v ose Z: průměrujeme pole přes přední a zadní čelo magnetu (z - L/2 až z + L/2)
    polovina_z_m = max(0.0005, mag_z_m / 2.0)
    Bx_predni = np.interp(zs_osa_m + polovina_z_m, zs_osa_m, Bx_osa)
    Bx_zadni = np.interp(zs_osa_m - polovina_z_m, zs_osa_m, Bx_osa)
    stredni_grad_Bx = (Bx_predni - Bx_zadni) / (2.0 * polovina_z_m)

    # Síla Fz(z) při 100% proudu v cívce
    Fz_plny_proud = m_dipol * stredni_grad_Bx

    # Aplikace polohy vypnutí cívky Z_cutoff v Záložce 1 (s reálným RL flyback doběhem za bodem vypnutí)
    z_off_mm = params["z_vypnuti_mm"]
    v_ref_ms = 2.5  # Referenční rychlost průletu 2.5 m/s (2.5 mm/ms) pro zobrazení RL doběhu po vypnutí
    Fz_aktivni = np.zeros_like(Fz_plny_proud)
    for idx_z, z_val in enumerate(zs_osa_mm):
        if z_val <= z_off_mm:
            Fz_aktivni[idx_z] = Fz_plny_proud[idx_z]
        else:
            # Po překročení Z_cutoff proud v cívce klesá s časovou konstantou tau_off_ms
            t_po_vypnuti_ms = (z_val - z_off_mm) / v_ref_ms
            podil_proudu = np.exp(-t_po_vypnuti_ms / max(0.02, tau_off_ms))
            Fz_aktivni[idx_z] = Fz_plny_proud[idx_z] * podil_proudu

    # Integrace práce W = integral(Fz dz) a impulzu síly (plocha pod křivkou)
    prace_aktivni_mJ = trapz_func(Fz_aktivni, zs_osa_m) * 1000.0
    integ_sila_Nmm = trapz_func(Fz_aktivni, zs_osa_mm)  # N*mm = mJ

    return {
        "zavity_1": zavity_1_civka,
        "r_stupne": r_stupne,
        "proud_ohm": proud_ohm,
        "proud_celkem": proud_celkem,
        "proud_omezen": proud_omezen,
        "proud_1_civka": proud_1_civka,
        "L_faze_uH": L_faze_uH,
        "tau_on_ms": tau_on_ms,
        "tau_off_ms": tau_off_ms,
        "v_flyback_spike": v_flyback_spike,
        "mosfet_safe": mosfet_safe,
        "xs_mm": xs_mm, "zs_mm": zs_mm,
        "Bx": Bx, "Bz": Bz, "B_intenzita": B_intenzita,
        "x_leve_celo_mm": x_leve_celo_mm, "x_prave_celo_mm": x_prave_celo_mm,
        "zs_osa_mm": zs_osa_mm,
        "Fz_osa": Fz_plny_proud,
        "Fz_aktivni": Fz_aktivni,
        "z_off_mm": z_off_mm,
        "prace_aktivni_mJ": prace_aktivni_mJ,
        "integ_sila_Nmm": integ_sila_Nmm,
        "f_peak_N": np.max(Fz_plny_proud)
    }

# =========================================================================
# 2. FYZIKÁLNÍ JÁDRO - ZÁLOŽKA 2: MULTI-STAGE KINEMATIKA & RL DYNAMIKA
# =========================================================================
def simuluj_prulet(faze_res, pocet_fazi, roztec_mm, m_celkem_g, mu_treni, casovani_ms):
    dprint(f"Tab2: Multi-stage průlet | Fází={pocet_fazi}, Rozteč={roztec_mm}mm, m_celkem={m_celkem_g}g")
    m_total_kg = max(0.001, m_celkem_g / 1000.0)
    g = 9.81
    f_treni_N = mu_treni * m_total_kg * g

    # Střed 1. cívky je na Z = 25 mm, vozíček startuje na Z = 13 mm (12 mm před středem 1. cívky)
    stredy_civek_mm = np.array([25.0 + i * roztec_mm for i in range(pocet_fazi)])
    delka_hlavne_mm = stredy_civek_mm[-1] + 25.0

    dt_ms = 0.05  # Krok 50 mikrosekund
    dt_s = dt_ms / 1000.0
    max_t_ms = 200.0
    kroky = int(max_t_ms / dt_ms)

    t_arr = np.zeros(kroky)
    z_arr = np.zeros(kroky)
    v_arr = np.zeros(kroky)
    led_state = np.zeros((kroky, pocet_fazi), dtype=int)  # 0=OFF, 1=GREEN(ON), 2=RED(FLYBACK)
    energie_fazi_mJ = np.zeros(pocet_fazi)
    suckback_fazi_mJ = np.zeros(pocet_fazi)  # Sleduje čistě brzdnou práci za středem cívky

    z_curr_mm = 13.0
    v_curr_ms = 0.0
    proudy_fazi = np.zeros(pocet_fazi)

    tau_on = max(0.02, faze_res["tau_on_ms"])
    tau_off = max(0.02, faze_res["tau_off_ms"])
    zs_lut = faze_res["zs_osa_mm"]
    fz_lut = faze_res["Fz_osa"]

    for k in range(kroky):
        t_ms = k * dt_ms
        t_arr[k] = t_ms
        z_arr[k] = z_curr_mm
        v_arr[k] = v_curr_ms

        f_mag_total = 0.0
        for i in range(pocet_fazi):
            t_on, trvani = casovani_ms[i]
            t_off = t_on + trvani

            if trvani > 0.01 and (t_on <= t_ms <= t_off):
                proudy_fazi[i] += (1.0 - proudy_fazi[i]) * (1.0 - np.exp(-dt_ms / tau_on))
                led_state[k, i] = 1
            else:
                proudy_fazi[i] *= np.exp(-dt_ms / tau_off)
                if proudy_fazi[i] > 0.05:
                    led_state[k, i] = 2
                else:
                    proudy_fazi[i] = 0.0
                    led_state[k, i] = 0

            if proudy_fazi[i] > 0.001:
                rel_z_mm = z_curr_mm - stredy_civek_mm[i]
                if -40.0 <= rel_z_mm <= 40.0:
                    f_i = np.interp(rel_z_mm, zs_lut, fz_lut) * proudy_fazi[i]
                    f_mag_total += f_i

        # Výpočet zrychlení a posunu (tření brzdí vždy proti směru rychlosti)
        if abs(v_curr_ms) > 1e-4:
            smer = 1.0 if v_curr_ms > 0 else -1.0
            f_net = f_mag_total - smer * f_treni_N
        else:
            if abs(f_mag_total) > f_treni_N:
                f_net = f_mag_total - np.sign(f_mag_total) * f_treni_N
            else:
                f_net = 0.0

        a_ms2 = f_net / m_total_kg
        v_next_ms = v_curr_ms + a_ms2 * dt_s

        # Zamezení kmitání kolem nuly kvůli tření, pokud nepůsobí magnetická síla
        if v_curr_ms * v_next_ms < 0 and abs(f_mag_total) <= f_treni_N:
            v_next_ms = 0.0

        # Zarážka na začátku hlavně (Z >= 5 mm)
        if z_curr_mm <= 5.0 and v_next_ms < 0.0:
            v_next_ms = 0.0

        dz_m = 0.5 * (v_curr_ms + v_next_ms) * dt_s

        # Přesný výpočet vykonané práce dW = F_i * dz pro každou cívku
        for i in range(pocet_fazi):
            if proudy_fazi[i] > 0.001:
                rel_z_mm = z_curr_mm - stredy_civek_mm[i]
                if -40.0 <= rel_z_mm <= 40.0:
                    f_i = np.interp(rel_z_mm, zs_lut, fz_lut) * proudy_fazi[i]
                    dw_mJ = (f_i * dz_m) * 1000.0
                    energie_fazi_mJ[i] += dw_mJ
                    if rel_z_mm > 0.5 and f_i < 0.0 and dz_m > 0.0:
                        suckback_fazi_mJ[i] += abs(dw_mJ)

        v_curr_ms = v_next_ms
        z_curr_mm = max(5.0, z_curr_mm + dz_m * 1000.0)

        if z_curr_mm >= delka_hlavne_mm:
            t_arr = t_arr[:k+1]
            z_arr = z_arr[:k+1]
            v_arr = v_arr[:k+1]
            led_state = led_state[:k+1, :]
            dprint(f"Tab2: Vozíček vyletěl z hlavně v t={t_ms:.2f}ms s rychlostí v_exit={v_curr_ms:.2f} m/s")
            break

    # Suck-back hlásíme pouze tehdy, když některá cívka reálně brzdila magnet za svým středem (> 0.5 mJ)
    has_suckback = bool(np.any(suckback_fazi_mJ > 0.5) or np.any(energie_fazi_mJ < -0.2))

    return {
        "t_arr": t_arr, "z_arr": z_arr, "v_arr": v_arr,
        "led_state": led_state, "energie_fazi_mJ": energie_fazi_mJ,
        "suckback_fazi_mJ": suckback_fazi_mJ, "has_suckback": has_suckback,
        "stredy_civek_mm": stredy_civek_mm, "delka_hlavne_mm": delka_hlavne_mm,
        "v_exit": max(0.0, v_arr[-1]), "v_max": np.max(v_arr)
    }

# =========================================================================
# 3. FYZIKÁLNÍ JÁDRO - ZÁLOŽKA 3: AERODYNAMIKA VLAŠTOVKY (0° AŽ 15°)
# =========================================================================
def spocitej_aerodynamiku(p3, v_muzzle_tab2):
    v0 = v_muzzle_tab2 if p3["use_tab2_speed"] else p3["v_manual"]
    v0 = max(0.5, v0)

    ld_ratio = max(0.5, p3["ld_ratio"])
    v_trim_ms = max(0.8, p3["v_trim_ms"])
    uhel_deg = np.clip(p3["uhel_hlavne_deg"], 0.0, 15.0)
    h0_m = max(0.1, p3["h_stolu_m"])

    g = 9.81
    gamma_trim = -np.arctan(1.0 / ld_ratio)
    cos_gt = np.cos(gamma_trim)
    sin_gt_abs = abs(np.sin(gamma_trim))

    dt = 0.005  # Krok 5 ms
    max_steps = 1600

    x_arr = np.zeros(max_steps)
    y_arr = np.zeros(max_steps)
    v_arr = np.zeros(max_steps)
    t_arr = np.zeros(max_steps)

    x, y = 0.0, h0_m
    v = v0
    gamma = np.radians(uhel_deg)
    stalled = False
    peak_y = h0_m
    k_damp = 1.65

    for k in range(max_steps):
        t_arr[k] = k * dt
        x_arr[k] = x
        y_arr[k] = y
        v_arr[k] = v
        peak_y = max(peak_y, y)

        v_ratio_sq = (v / v_trim_ms) ** 2
        lift_acc = g * cos_gt * v_ratio_sq
        drag_acc = g * sin_gt_abs * v_ratio_sq

        dv_dt = -drag_acc - g * np.sin(gamma)
        dgamma_dt = (lift_acc - g * np.cos(gamma)) / max(0.5, v) - k_damp * (gamma - gamma_trim)

        v = max(0.2, v + dv_dt * dt)
        gamma = gamma + dgamma_dt * dt

        if np.degrees(gamma) > 50.0 or (v < 0.62 * v_trim_ms and y > h0_m + 0.15):
            stalled = True
            gamma = max(np.radians(-65.0), gamma - 4.5 * dt)
            v = max(0.5, v - 1.5 * dt)

        x += v * np.cos(gamma) * dt
        y += v * np.sin(gamma) * dt

        if y <= 0.0:
            y_arr[k] = 0.0
            x_arr, y_arr, v_arr, t_arr = x_arr[:k+1], y_arr[:k+1], v_arr[:k+1], t_arr[:k+1]
            break

    dprint(f"Tab3: v0={v0:.2f}m/s, v_trim={v_trim_ms:.2f}m/s, L/D={ld_ratio:.1f}, Úhel={uhel_deg:.1f}°, Dolet={x_arr[-1]:.2f}m")

    return {
        "v0": v0,
        "ld_ratio": ld_ratio,
        "v_trim_ms": v_trim_ms,
        "uhel_deg": uhel_deg,
        "x_tr": x_arr, "y_tr": y_arr, "v_tr": v_arr, "t_tr": t_arr,
        "dolet_m": x_arr[-1],
        "peak_y": peak_y,
        "stalled": stalled
    }

# =========================================================================
# 4. HLAVNÍ GUI APLIKACE (3 ZÁLOŽKY, DARK MODE + MCLAREN PAPAYA ORANGE)
# =========================================================================
class ExpediceMarsSimV5:
    def __init__(self, root):
        self.root = root
        self.root.title("🏎️ Expedice Mars | Papaya Telemetry & Coilgun Engineering Suite v5.0")
        self.root.geometry("1520x920")
        self.root.configure(bg=BG_DARK)

        self._nastav_papaya_dark_styl()

        self.notebook = ttk.Notebook(root)
        self.notebook.pack(fill=tk.BOTH, expand=True, padx=6, pady=6)

        self.tab1 = ttk.Frame(self.notebook, padding=8)
        self.tab2 = ttk.Frame(self.notebook, padding=8)
        self.tab3 = ttk.Frame(self.notebook, padding=8)

        self.notebook.add(self.tab1, text="⚡ 1. Elektromagnetismus & XY-MOS Laboratoř")
        self.notebook.add(self.tab2, text="🚀 2. Multi-Stage Kinematika, Časování & High-Speed Kamera")
        self.notebook.add(self.tab3, text="✈️ 3. Aerodynamika Vlaštovky & Úhel Výstřelu")

        # Výchozí parametry Tab 1 (Hranol 10x10x10 mm, realistická síla)
        self.p1 = {
            "napeti": 12.0,
            "proud_limit": 25.0,
            "pocet_fazi": 6,
            "d_vnitrni": 6.5,
            "d_vnejsi": 25.0,
            "sirka": 4.0,
            "d_dratu": 0.4,
            "delka_dratu": 7.5,
            "mezera": 12.0,
            "zapojeni": "Paralelně",
            "r_stupne": 0.55,
            "mag_x": 10.0,
            "mag_y": 10.0,
            "mag_z": 10.0,
            "mag_grade": "N42",
            "z_vypnuti_mm": -2.0
        }

        # Výchozí parametry Tab 2 (Sjednocená celková hmotnost)
        self.p2 = {
            "m_celkem_g": 15.0,
            "roztec_mm": 27.0,
            "mu_treni": 0.20
        }

        # Výchozí parametry Tab 3 (Přímé zadání L/D a v_trim, úhel 0-15°)
        self.p3 = {
            "use_tab2_speed": True,
            "v_manual": 5.0,
            "ld_ratio": 5.0,
            "v_trim_ms": 3.5,
            "uhel_hlavne_deg": 10.0,
            "h_stolu_m": 0.85
        }

        # Sdílená proměnná počtu stupňů mezi Tab 1 a Tab 2
        self.pocet_fazi_var = tk.DoubleVar(value=self.p1["pocet_fazi"])

        self.casovani_vars = []
        default_timings = [
            (0.0, 10.0), (11.0, 7.0), (19.0, 5.5), (25.5, 4.5), (31.0, 4.0),
            (36.0, 3.5), (40.5, 3.5), (44.5, 3.0), (48.0, 3.0), (51.5, 3.0)
        ]
        for t_on, dt_on in default_timings:
            self.casovani_vars.append((tk.DoubleVar(value=t_on), tk.DoubleVar(value=dt_on)))

        self.anim_running = False
        self.anim_frame_idx = 0

        self._postav_tab1()
        self._postav_tab2()
        self._postav_tab3()
        self.aktualizuj_vse()

    def _nastav_papaya_dark_styl(self):
        style = ttk.Style()
        style.theme_use("clam")

        style.configure(".", background=BG_DARK, foreground=FG_TEXT, fieldbackground=BG_INPUT, font=("Segoe UI", 9))
        style.configure("TFrame", background=BG_DARK)
        style.configure("Panel.TFrame", background=BG_PANEL)
        style.configure("TLabel", background=BG_DARK, foreground=FG_TEXT)
        style.configure("Panel.TLabel", background=BG_PANEL, foreground=FG_TEXT)
        style.configure("Muted.TLabel", background=BG_PANEL, foreground=FG_MUTED, font=("Segoe UI", 8, "italic"))
        style.configure("Header.TLabel", background=BG_PANEL, foreground=PAPAYA, font=("Segoe UI", 10, "bold"))

        style.configure("TNotebook", background=BG_DARK, borderwidth=0)
        style.configure("TNotebook.Tab", background=BG_PANEL, foreground=FG_MUTED, padding=[14, 6], font=("Segoe UI", 10, "bold"))
        style.map("TNotebook.Tab", background=[("selected", PAPAYA)], foreground=[("selected", "#000000")])

        style.configure("TButton", background=PAPAYA, foreground="#000000", font=("Segoe UI", 9, "bold"), borderwidth=0, padding=6)
        style.map("TButton", background=[("active", PAPAYA_LIGHT)])

        style.configure("Secondary.TButton", background="#2F3542", foreground=FG_TEXT, font=("Segoe UI", 9, "bold"), padding=5)
        style.map("Secondary.TButton", background=[("active", "#57606F")])

        style.configure("TEntry", fieldbackground=BG_INPUT, foreground=PAPAYA_LIGHT, insertcolor=FG_TEXT, borderwidth=1)
        style.configure("TCombobox", fieldbackground=BG_INPUT, background=BG_PANEL, foreground=PAPAYA_LIGHT)
        style.map("TCombobox", fieldbackground=[("readonly", BG_INPUT)], foreground=[("readonly", PAPAYA_LIGHT)])
        style.configure("TSeparator", background=GRID_COL)
        style.configure("Horizontal.TScale", background=BG_PANEL, troughcolor=BG_INPUT)

    def _styluj_graf(self, ax):
        ax.set_facecolor(BG_PANEL)
        ax.tick_params(colors=FG_MUTED, labelsize=8)
        ax.xaxis.label.set_color(FG_TEXT)
        ax.yaxis.label.set_color(FG_TEXT)
        ax.title.set_color(PAPAYA)
        for spine in ax.spines.values():
            spine.set_color(GRID_COL)

    # ---------------------------------------------------------------------
    # STAVBA ZÁLOŽKY 1: ELEKTROMAGNETISMUS & XY-MOS
    # ---------------------------------------------------------------------
    def _postav_tab1(self):
        left = ttk.Frame(self.tab1, style="Panel.TFrame", width=395, padding=10)
        left.pack(side=tk.LEFT, fill=tk.Y, padx=(0, 8))
        left.pack_propagate(False)

        right = ttk.Frame(self.tab1)
        right.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)

        ttk.Label(left, text="1. GEOMETRIE, ZDROJ & PROUDOVÝ LIMIT", style="Header.TLabel").pack(anchor=tk.W, pady=(0, 4))

        self.vars1 = {}
        polozky_1 = [
            ("napeti", "Napětí baterie (V) [3S LiPo ≈ 12V]:"),
            ("proud_limit", "Proudové omezení zdroje / BMS I_max (A):"),
            ("pocet_fazi", "Počet stupňů (fází) za sebou (1-10):"),
            ("d_vnitrni", "Vnitřní průměr cívky d_in (mm):"),
            ("d_vnejsi", "Vnější průměr cívky d_out (mm):"),
            ("sirka", "Tloušťka vinutí cívky (mm):"),
            ("d_dratu", "Průměr CuL drátu (mm):"),
            ("delka_dratu", "Délka drátu na 1 cívce (m):"),
            ("mezera", "Vzdálenost mezi cívkami / šířka koryta (mm):")
        ]

        for key, label in polozky_1:
            row = ttk.Frame(left, style="Panel.TFrame", padding=1)
            row.pack(fill=tk.X)
            ttk.Label(row, text=label, style="Panel.TLabel", font=("Segoe UI", 8)).pack(side=tk.LEFT)
            var = self.pocet_fazi_var if key == "pocet_fazi" else tk.DoubleVar(value=self.p1[key])
            self.vars1[key] = var
            ent = ttk.Entry(row, textvariable=var, width=8)
            ent.pack(side=tk.RIGHT)
            ent.bind("<Return>", lambda e: self.aktualizuj_vse())
            ent.bind("<FocusOut>", lambda e: self.aktualizuj_vse())

        ttk.Separator(left, orient="horizontal").pack(fill=tk.X, pady=6)
        ttk.Label(left, text="2. ODPOR & ZAPOJENÍ STUPNĚ (XY-MOS)", style="Header.TLabel").pack(anchor=tk.W, pady=(0, 3))

        row_zap = ttk.Frame(left, style="Panel.TFrame", padding=2)
        row_zap.pack(fill=tk.X)
        ttk.Label(row_zap, text="Zapojení 2 cívek v 1 stupni:", style="Panel.TLabel", font=("Segoe UI", 8, "bold")).pack(side=tk.LEFT)
        self.zapojeni_var = tk.StringVar(value=self.p1["zapojeni"])
        cb_zap = ttk.Combobox(row_zap, textvariable=self.zapojeni_var, values=["Sériově", "Paralelně"], state="readonly", width=10)
        cb_zap.pack(side=tk.RIGHT)
        cb_zap.bind("<<ComboboxSelected>>", lambda e: self.aktualizuj_vse())

        row_rst = ttk.Frame(left, style="Panel.TFrame", padding=2)
        row_rst.pack(fill=tk.X)
        ttk.Label(row_rst, text="Celkový odpor jednoho stupně R_stage (Ω):", style="Panel.TLabel", font=("Segoe UI", 8, "bold")).pack(side=tk.LEFT)
        self.vars1["r_stupne"] = tk.DoubleVar(value=self.p1["r_stupne"])
        ent_rst = ttk.Entry(row_rst, textvariable=self.vars1["r_stupne"], width=8)
        ent_rst.pack(side=tk.RIGHT)
        ent_rst.bind("<Return>", lambda e: self.aktualizuj_vse())
        ent_rst.bind("<FocusOut>", lambda e: self.aktualizuj_vse())

        ttk.Separator(left, orient="horizontal").pack(fill=tk.X, pady=6)
        ttk.Label(left, text="3. HRANOLOVÝ MAGNET X × Y × Z (mm)", style="Header.TLabel").pack(anchor=tk.W, pady=(0, 3))

        polozky_mag = [
            ("mag_x", "Šířka magnetu X napříč korytem (mm):"),
            ("mag_y", "Výška magnetu Y svisle (mm):"),
            ("mag_z", "Délka magnetu Z ve směru jízdy (mm):")
        ]
        for key, label in polozky_mag:
            row = ttk.Frame(left, style="Panel.TFrame", padding=1)
            row.pack(fill=tk.X)
            ttk.Label(row, text=label, style="Panel.TLabel", font=("Segoe UI", 8)).pack(side=tk.LEFT)
            var = tk.DoubleVar(value=self.p1[key])
            self.vars1[key] = var
            ent = ttk.Entry(row, textvariable=var, width=8)
            ent.pack(side=tk.RIGHT)
            ent.bind("<Return>", lambda e: self.aktualizuj_vse())
            ent.bind("<FocusOut>", lambda e: self.aktualizuj_vse())

        row_grade = ttk.Frame(left, style="Panel.TFrame", padding=1)
        row_grade.pack(fill=tk.X)
        ttk.Label(row_grade, text="Třída neodymu:", style="Panel.TLabel", font=("Segoe UI", 8)).pack(side=tk.LEFT)
        self.mag_grade_var = tk.StringVar(value=self.p1["mag_grade"])
        cb_grade = ttk.Combobox(row_grade, textvariable=self.mag_grade_var, values=["N35", "N38", "N42", "N52"], state="readonly", width=8)
        cb_grade.pack(side=tk.RIGHT)
        cb_grade.bind("<<ComboboxSelected>>", lambda e: self.aktualizuj_vse())

        ttk.Separator(left, orient="horizontal").pack(fill=tk.X, pady=6)
        ttk.Label(left, text="4. OKAMŽIK VYPNUTÍ CÍVKY & INTEGRACE SÍLY", style="Header.TLabel").pack(anchor=tk.W, pady=(0, 2))

        row_zoff = ttk.Frame(left, style="Panel.TFrame", padding=1)
        row_zoff.pack(fill=tk.X)
        ttk.Label(row_zoff, text="Vypnout cívku na pozici Z_cutoff (mm):", style="Panel.TLabel", font=("Segoe UI", 8, "bold")).pack(side=tk.LEFT)
        self.vars1["z_vypnuti_mm"] = tk.DoubleVar(value=self.p1["z_vypnuti_mm"])
        ent_zoff = ttk.Entry(row_zoff, textvariable=self.vars1["z_vypnuti_mm"], width=8)
        ent_zoff.pack(side=tk.RIGHT)
        ent_zoff.bind("<Return>", lambda e: self.aktualizuj_vse())

        self.zoff_slider = ttk.Scale(
            left, from_=-20.0, to=20.0, variable=self.vars1["z_vypnuti_mm"],
            command=lambda val: self._on_zoff_slider(val)
        )
        self.zoff_slider.pack(fill=tk.X, pady=3)

        self.integ_info_lbl = tk.Label(
            left, text="", bg="#262933", fg=PAPAYA_LIGHT,
            font=("Consolas", 8, "bold"), wraplength=365, justify=tk.LEFT, padx=8, pady=5
        )
        self.integ_info_lbl.pack(fill=tk.X, pady=3)

        ttk.Button(left, text="⚡ PŘEPOČÍTAT ELEKTROMAGNETISMUS", command=self.aktualizuj_vse).pack(fill=tk.X, pady=6)

        self.status_banner1 = tk.Label(right, text="", font=("Consolas", 10, "bold"), justify=tk.LEFT, anchor=tk.W, padx=12, pady=8)
        self.status_banner1.pack(side=tk.TOP, fill=tk.X, pady=(0, 6))

        self.fig1, (self.ax_flux, self.ax_force) = plt.subplots(2, 1, figsize=(8, 6.8), facecolor=BG_DARK)
        self.fig1.tight_layout(pad=3.2)
        self.canvas1 = FigureCanvasTkAgg(self.fig1, master=right)
        self.canvas1.get_tk_widget().pack(side=tk.BOTTOM, fill=tk.BOTH, expand=True)

    def _on_zoff_slider(self, val):
        rounded = round(float(val) * 2.0) / 2.0
        self.vars1["z_vypnuti_mm"].set(rounded)
        self.aktualizuj_vse()

    # ---------------------------------------------------------------------
    # STAVBA ZÁLOŽKY 2: MULTI-STAGE KINEMATIKA, ČASOVÁNÍ & ANIMACE
    # ---------------------------------------------------------------------
    def _postav_tab2(self):
        left = ttk.Frame(self.tab2, style="Panel.TFrame", width=410, padding=10)
        left.pack(side=tk.LEFT, fill=tk.Y, padx=(0, 8))
        left.pack_propagate(False)

        right = ttk.Frame(self.tab2)
        right.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)

        ttk.Label(left, text="1. POČET STUPŇŮ, HMOTNOST & DRÁHA", style="Header.TLabel").pack(anchor=tk.W, pady=(0, 4))

        row_nf = ttk.Frame(left, style="Panel.TFrame", padding=2)
        row_nf.pack(fill=tk.X)
        ttk.Label(row_nf, text="Počet stupňů (fází) za sebou (1-10):", style="Panel.TLabel", font=("Segoe UI", 8, "bold")).pack(side=tk.LEFT)
        ent_nf = ttk.Entry(row_nf, textvariable=self.pocet_fazi_var, width=7)
        ent_nf.pack(side=tk.RIGHT)
        ent_nf.bind("<Return>", lambda e: self.aktualizuj_vse())
        ent_nf.bind("<FocusOut>", lambda e: self.aktualizuj_vse())

        self.vars2 = {}
        polozky_2 = [
            ("m_celkem_g", "Celková hmotnost (vozíček + magnet + vlaštovka) [g]:"),
            ("roztec_mm", "Vzdálenost středů stupňů (mm) [min = d_out]:"),
            ("mu_treni", "Koeficient tření 3D tisku μ (běžně 0.18 - 0.25):")
        ]
        for key, label in polozky_2:
            row = ttk.Frame(left, style="Panel.TFrame", padding=2)
            row.pack(fill=tk.X)
            ttk.Label(row, text=label, style="Panel.TLabel", font=("Segoe UI", 8)).pack(side=tk.LEFT)
            var = tk.DoubleVar(value=self.p2[key])
            self.vars2[key] = var
            ent = ttk.Entry(row, textvariable=var, width=7)
            ent.pack(side=tk.RIGHT)
            ent.bind("<Return>", lambda e: self.aktualizuj_vse())
            ent.bind("<FocusOut>", lambda e: self.aktualizuj_vse())

        ttk.Separator(left, orient="horizontal").pack(fill=tk.X, pady=8)
        ttk.Label(left, text="2. MANUÁLNÍ ČASOVÁNÍ SEPNUTÍ STUPŇŮ (ms)", style="Header.TLabel").pack(anchor=tk.W)
        ttk.Label(left, text="Pozdní vypnutí za středem cívky = ČERVENÁ flyback LED magnet brzdí (Suck-back)!", style="Muted.TLabel", wraplength=385).pack(anchor=tk.W, pady=(0, 4))

        self.timing_rows_frame = ttk.Frame(left, style="Panel.TFrame")
        self.timing_rows_frame.pack(fill=tk.X)
        self.stage_row_widgets = []

        hdr = ttk.Frame(self.timing_rows_frame, style="Panel.TFrame")
        hdr.pack(fill=tk.X)
        ttk.Label(hdr, text="Stupeň", width=9, style="Panel.TLabel", font=("Segoe UI", 8, "bold")).pack(side=tk.LEFT)
        ttk.Label(hdr, text="Zapnout t_on (ms)", width=16, style="Panel.TLabel", font=("Segoe UI", 8, "bold")).pack(side=tk.LEFT)
        ttk.Label(hdr, text="Doba sepnutí Δt (ms)", width=18, style="Panel.TLabel", font=("Segoe UI", 8, "bold")).pack(side=tk.LEFT)

        for i in range(10):
            rframe = ttk.Frame(self.timing_rows_frame, style="Panel.TFrame", padding=1)
            rframe.pack(fill=tk.X)
            ttk.Label(rframe, text=f"Cívka #{i+1}:", width=9, style="Panel.TLabel", font=("Segoe UI", 8, "bold")).pack(side=tk.LEFT)
            t_on_var, dt_var = self.casovani_vars[i]

            e_on = ttk.Entry(rframe, textvariable=t_on_var, width=10)
            e_on.pack(side=tk.LEFT, padx=8)
            e_on.bind("<Return>", lambda e: self.aktualizuj_vse())
            e_on.bind("<FocusOut>", lambda e: self.aktualizuj_vse())

            e_dt = ttk.Entry(rframe, textvariable=dt_var, width=10)
            e_dt.pack(side=tk.LEFT, padx=8)
            e_dt.bind("<Return>", lambda e: self.aktualizuj_vse())
            e_dt.bind("<FocusOut>", lambda e: self.aktualizuj_vse())

            self.stage_row_widgets.append(rframe)

        btn_frame = ttk.Frame(left, style="Panel.TFrame", padding=4)
        btn_frame.pack(fill=tk.X, pady=6)
        ttk.Button(btn_frame, text="🚀 SPUSTIT SIMULACI PRŮLETU", command=self.aktualizuj_vse).pack(fill=tk.X, pady=2)
        ttk.Button(btn_frame, text="🎬 PŘEHRÁT / ZASTAVIT ANIMACI S LEDKAMI", style="Secondary.TButton", command=self.toggle_animace).pack(fill=tk.X, pady=2)

        if ADMIN_AUTO_OPTIMIZE:
            ttk.Button(btn_frame, text="🎯 AUTO-OPTIMALIZACE ČASŮ (ADMIN)", style="Secondary.TButton", command=self.admin_najdi_optimum).pack(fill=tk.X, pady=2)

        ttk.Label(left, text="Ruční posuvník High-Speed Kamery (čas ms):", style="Panel.TLabel", font=("Segoe UI", 8, "bold")).pack(anchor=tk.W, pady=(6, 0))
        self.scrub_var = tk.DoubleVar(value=0.0)
        self.scrub_slider = ttk.Scale(left, from_=0.0, to=100.0, variable=self.scrub_var, command=self.on_scrub_move)
        self.scrub_slider.pack(fill=tk.X, pady=4)

        self.status_banner2 = tk.Label(right, text="", font=("Consolas", 10, "bold"), bg=BG_PANEL, fg=PAPAYA, justify=tk.LEFT, anchor=tk.W, padx=12, pady=8)
        self.status_banner2.pack(side=tk.TOP, fill=tk.X, pady=(0, 6))

        self.fig2, (self.ax_anim, self.ax_vel, self.ax_energy) = plt.subplots(
            3, 1, figsize=(8.5, 7.2), facecolor=BG_DARK, gridspec_kw={'height_ratios': [1.1, 1.3, 1.0]}
        )
        self.fig2.tight_layout(pad=2.8)
        self.canvas2 = FigureCanvasTkAgg(self.fig2, master=right)
        self.canvas2.get_tk_widget().pack(side=tk.BOTTOM, fill=tk.BOTH, expand=True)

    # ---------------------------------------------------------------------
    # STAVBA ZÁLOŽKY 3: AERODYNAMIKA VLAŠTOVKY (0° AŽ 15°)
    # ---------------------------------------------------------------------
    def _postav_tab3(self):
        left = ttk.Frame(self.tab3, style="Panel.TFrame", width=395, padding=10)
        left.pack(side=tk.LEFT, fill=tk.Y, padx=(0, 8))
        left.pack_propagate(False)

        right = ttk.Frame(self.tab3)
        right.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)

        ttk.Label(left, text="1. VÝSTUPNÍ RYCHLOST Z KATAPULTU", style="Header.TLabel").pack(anchor=tk.W, pady=(0, 4))

        self.use_tab2_var = tk.BooleanVar(value=self.p3["use_tab2_speed"])
        chk = tk.Checkbutton(
            left, text="Přebírat výstupní rychlost automaticky ze Záložky 2",
            variable=self.use_tab2_var, bg=BG_PANEL, fg=FG_TEXT, selectcolor=BG_INPUT,
            activebackground=BG_PANEL, activeforeground=PAPAYA, command=self.aktualizuj_vse
        )
        chk.pack(anchor=tk.W, pady=2)

        self.vars3 = {}
        row_vm = ttk.Frame(left, style="Panel.TFrame", padding=2)
        row_vm.pack(fill=tk.X)
        ttk.Label(row_vm, text="Ruční rychlost výstřelu v_0 (m/s) [pokud ne ze Zál. 2]:", style="Panel.TLabel", font=("Segoe UI", 8)).pack(side=tk.LEFT)
        self.vars3["v_manual"] = tk.DoubleVar(value=self.p3["v_manual"])
        ent_vm = ttk.Entry(row_vm, textvariable=self.vars3["v_manual"], width=7)
        ent_vm.pack(side=tk.RIGHT)
        ent_vm.bind("<Return>", lambda e: self.aktualizuj_vse())
        ent_vm.bind("<FocusOut>", lambda e: self.aktualizuj_vse())

        ttk.Separator(left, orient="horizontal").pack(fill=tk.X, pady=8)
        ttk.Label(left, text="2. PARAMETRY VLAŠTOVKY (SPOČÍTANÉ NA PAPÍŘE)", style="Header.TLabel").pack(anchor=tk.W, pady=(0, 4))

        polozky_aero = [
            ("ld_ratio", "Klouzavost vlaštovky L/D [-]:"),
            ("v_trim_ms", "Rovnovážná klouzavá rychlost v_trim (m/s):")
        ]
        for key, label in polozky_aero:
            row = ttk.Frame(left, style="Panel.TFrame", padding=2)
            row.pack(fill=tk.X)
            ttk.Label(row, text=label, style="Panel.TLabel", font=("Segoe UI", 8)).pack(side=tk.LEFT)
            var = tk.DoubleVar(value=self.p3[key])
            self.vars3[key] = var
            ent = ttk.Entry(row, textvariable=var, width=7)
            ent.pack(side=tk.RIGHT)
            ent.bind("<Return>", lambda e: self.aktualizuj_vse())
            ent.bind("<FocusOut>", lambda e: self.aktualizuj_vse())

        ttk.Separator(left, orient="horizontal").pack(fill=tk.X, pady=8)
        ttk.Label(left, text="3. NÁKLON HLAVNĚ KATAPULTU (0° AŽ 15°)", style="Header.TLabel").pack(anchor=tk.W, pady=(0, 4))

        polozky_uhel = [
            ("uhel_hlavne_deg", "Úhel náklonu hlavně θ_gun (°) [0.0 - 15.0°]:"),
            ("h_stolu_m", "Výška ústí hlavně nad podlahou h_0 (m):")
        ]
        for key, label in polozky_uhel:
            row = ttk.Frame(left, style="Panel.TFrame", padding=2)
            row.pack(fill=tk.X)
            ttk.Label(row, text=label, style="Panel.TLabel", font=("Segoe UI", 8)).pack(side=tk.LEFT)
            var = tk.DoubleVar(value=self.p3[key])
            self.vars3[key] = var
            ent = ttk.Entry(row, textvariable=var, width=7)
            ent.pack(side=tk.RIGHT)
            ent.bind("<Return>", lambda e: self.aktualizuj_vse())
            ent.bind("<FocusOut>", lambda e: self.aktualizuj_vse())

        ttk.Label(left, text="Posuvník úhlu náklonu hlavně θ_gun (0° - 15°):", style="Panel.TLabel", font=("Segoe UI", 8, "bold")).pack(anchor=tk.W, pady=(6, 0))
        self.uhel_slider = ttk.Scale(
            left, from_=0.0, to=15.0, variable=self.vars3["uhel_hlavne_deg"],
            command=lambda val: self._on_uhel_slider(val)
        )
        self.uhel_slider.pack(fill=tk.X, pady=4)

        ttk.Button(left, text="✈️ SPOČÍTAT TRAJEKTORII LETU", command=self.aktualizuj_vse).pack(fill=tk.X, pady=10)

        self.status_banner3 = tk.Label(right, text="", font=("Consolas", 10, "bold"), bg=BG_PANEL, fg=PAPAYA, justify=tk.LEFT, anchor=tk.W, padx=12, pady=8)
        self.status_banner3.pack(side=tk.TOP, fill=tk.X, pady=(0, 6))

        self.fig3, self.ax_traj = plt.subplots(1, 1, figsize=(8.5, 6.8), facecolor=BG_DARK)
        self.fig3.tight_layout(pad=3.2)
        self.canvas3 = FigureCanvasTkAgg(self.fig3, master=right)
        self.canvas3.get_tk_widget().pack(side=tk.BOTTOM, fill=tk.BOTH, expand=True)

    def _on_uhel_slider(self, val):
        rounded = round(float(val) * 2.0) / 2.0
        self.vars3["uhel_hlavne_deg"].set(rounded)
        self.aktualizuj_vse()

    # ---------------------------------------------------------------------
    # HLAVNÍ AKTUALIZAČNÍ SMYČKA VŠECH 3 ZÁLOŽEK
    # ---------------------------------------------------------------------
    def _nacti_vstupy(self):
        for k, var in self.vars1.items():
            try:
                self.p1[k] = float(var.get())
            except ValueError:
                pass
        n_stag = int(max(1, min(10, round(self.pocet_fazi_var.get()))))
        self.p1["pocet_fazi"] = n_stag
        self.pocet_fazi_var.set(n_stag)
        self.p1["zapojeni"] = self.zapojeni_var.get()
        self.p1["mag_grade"] = self.mag_grade_var.get()
        self.p1["z_vypnuti_mm"] = float(np.clip(self.p1["z_vypnuti_mm"], -35.0, 35.0))

        for k, var in self.vars2.items():
            try:
                self.p2[k] = float(var.get())
            except ValueError:
                pass

        min_roztec = self.p1["d_vnejsi"] + 1.0
        if self.p2["roztec_mm"] < min_roztec:
            self.p2["roztec_mm"] = min_roztec
            self.vars2["roztec_mm"].set(min_roztec)

        self.p3["use_tab2_speed"] = bool(self.use_tab2_var.get())
        for k, var in self.vars3.items():
            try:
                self.p3[k] = float(var.get())
            except ValueError:
                pass
        self.p3["uhel_hlavne_deg"] = float(np.clip(self.p3["uhel_hlavne_deg"], 0.0, 15.0))
        self.vars3["uhel_hlavne_deg"].set(self.p3["uhel_hlavne_deg"])

    def aktualizuj_vse(self):
        self._nacti_vstupy()
        self.faze_res = spocitej_fazi(self.p1)

        n_fazi = self.p1["pocet_fazi"]
        for i, w in enumerate(self.stage_row_widgets):
            if i < n_fazi:
                w.pack(fill=tk.X)
            else:
                w.pack_forget()

        res = self.faze_res

        self.integ_info_lbl.config(
            text=f"Vypnutí na Z_cutoff = {res['z_off_mm']:.1f} mm:\n"
                 f"• Špičková síla F_max = {res['f_peak_N']:.2f} N\n"
                 f"• Integrovaná práce ∫Fz dz = {res['prace_aktivni_mJ']:.2f} mJ ({res['integ_sila_Nmm']:.2f} N·mm)"
        )

        mosfet_txt = "✅ XY-MOS V BEZPEČÍ (< 40 V)" if res["mosfet_safe"] else "💥 PRŮRAZ XY-MOSFETU (> 40 V PŘI VYPNUTÍ)!"
        lim_txt = f" [OMEZENO Z {res['proud_ohm']:.1f} A]" if res["proud_omezen"] else ""
        banner_bg = BG_PANEL if res["mosfet_safe"] else RED_WARN
        banner_fg = PAPAYA if res["mosfet_safe"] else "#FFFFFF"

        txt1 = (
            f"CÍVKA: ~{res['zavity_1']} závitů | R_stupně ({self.p1['zapojeni']}) = {res['r_stupne']:.2f} Ω | Indukčnost L_fáze = {res['L_faze_uH']:.0f} µH\n"
            f"PROUD & FLYBACK: Proud = {res['proud_celkem']:.1f} A{lim_txt} ({res['proud_1_civka']:.1f} A/cívka) | Flyback = {res['v_flyback_spike']:.1f} V --> {mosfet_txt}\n"
            f"DYNAMIKA & PRÁCE: Náběh Tau_on = {res['tau_on_ms']:.2f} ms | Vybíjení Tau_off = {res['tau_off_ms']:.2f} ms | Práce stupně (do Z={res['z_off_mm']:.1f}mm) = {res['prace_aktivni_mJ']:.1f} mJ"
        )
        self.status_banner1.config(text=txt1, bg=banner_bg, fg=banner_fg)
        self._vykresli_tab1()

        # Tab 2 výpočet
        casovani_list = []
        for i in range(n_fazi):
            try:
                t_on = float(self.casovani_vars[i][0].get())
                dt_on = float(self.casovani_vars[i][1].get())
            except ValueError:
                t_on, dt_on = 0.0, 5.0
            casovani_list.append((t_on, dt_on))

        self.sim_res = simuluj_prulet(
            self.faze_res, n_fazi, self.p2["roztec_mm"],
            self.p2["m_celkem_g"], self.p2["mu_treni"], casovani_list
        )
        self.scrub_slider.config(to=float(self.sim_res["t_arr"][-1]))
        self._vykresli_tab2()

        # Tab 3 výpočet
        self.aero_res = spocitej_aerodynamiku(self.p3, self.sim_res["v_exit"])
        self._vykresli_tab3()

    def _vykresli_tab1(self):
        res = self.faze_res
        self.ax_flux.clear()
        self._styluj_graf(self.ax_flux)
        self.ax_flux.imshow(res["B_intenzita"], extent=[-25, 25, -25, 25], origin="lower", cmap="inferno", aspect="auto")
        self.ax_flux.streamplot(res["xs_mm"], res["zs_mm"], res["Bx"], res["Bz"], color="#F5F6FA", density=0.85, linewidth=0.55, arrowsize=0.8)

        w = self.p1["sirka"]
        r_in = self.p1["d_vnitrni"] / 2.0
        r_out = self.p1["d_vnejsi"] / 2.0
        h = r_out - r_in
        xl, xp = res["x_leve_celo_mm"], res["x_prave_celo_mm"]

        for x_start in [xl - w, xp]:
            self.ax_flux.add_patch(Rectangle((x_start, r_in), w, h, fill=False, edgecolor=PAPAYA, lw=2))
            self.ax_flux.add_patch(Rectangle((x_start, -r_out), w, h, fill=False, edgecolor=PAPAYA, lw=2))

        self.ax_flux.axvline(x=xl, color=CYAN_ACCENT, linestyle="--", lw=1)
        self.ax_flux.axvline(x=xp, color=CYAN_ACCENT, linestyle="--", lw=1)
        self.ax_flux.set_title("2D Řez magnetickým tokem B [T] (X = příčná šířka koryta [mm], Z = podélná osa jízdy [mm])", fontsize=9, fontweight="bold")
        self.ax_flux.set_xlabel("Příčná osa koryta X [mm]")
        self.ax_flux.set_ylabel("Podélná osa dráhy Z [mm]")
        self.ax_flux.set_xlim(-25, 25)
        self.ax_flux.set_ylim(-25, 25)

        self.ax_force.clear()
        self._styluj_graf(self.ax_force)
        zs = res["zs_osa_mm"]
        fz_full = res["Fz_osa"]
        fz_act = res["Fz_aktivni"]

        self.ax_force.plot(zs, fz_full, color=FG_MUTED, linestyle=":", lw=1.4, label="Statická síla Fz při trvalém proudu [N]")
        self.ax_force.plot(zs, fz_act, color=PAPAYA, lw=2.5, label=f"Aktivní síla Fz (vypnutí Z = {res['z_off_mm']:.1f} mm) [N]")
        self.ax_force.fill_between(zs, 0, fz_act, where=(fz_act >= 0), color=GREEN_OK, alpha=0.25, label="Urychlovací práce (+mJ)")
        self.ax_force.fill_between(zs, 0, fz_act, where=(fz_act < 0), color=RED_WARN, alpha=0.30, label="Suck-back brzdná práce (-mJ)")
        self.ax_force.axvline(x=res["z_off_mm"], color=CYAN_ACCENT, linestyle="--", lw=1.8, label=f"Bod vypnutí Z_cutoff = {res['z_off_mm']:.1f} mm")
        self.ax_force.axvline(x=0, color=FG_MUTED, linestyle="-", lw=0.9)
        self.ax_force.axhline(y=0, color=FG_MUTED, lw=0.8)

        self.ax_force.set_title(
            f"Podélná magnetická síla Fz [N] | Integrovaná práce ∫Fz dz = {res['prace_aktivni_mJ']:.2f} mJ ({res['integ_sila_Nmm']:.2f} N·mm)",
            fontsize=9, fontweight="bold"
        )
        self.ax_force.set_xlabel("Poloha středu magnetu vůči středu cívky Z [mm] (0 mm = střed cívky)")
        self.ax_force.set_ylabel("Podélná síla Fz [N]")
        self.ax_force.legend(loc="upper right", fontsize=8, facecolor=BG_DARK, edgecolor=GRID_COL, labelcolor=FG_TEXT)
        self.ax_force.grid(True, color=GRID_COL, alpha=0.5)

        self.canvas1.draw()

    def _vykresli_tab2(self):
        sr = self.sim_res
        m_tot_kg = self.p2["m_celkem_g"] / 1000.0
        ek_mJ = 0.5 * m_tot_kg * (sr["v_exit"] ** 2) * 1000.0
        has_suckback = sr["has_suckback"]
        suckback_txt = "⚠️ DETEKOVÁN SUCK-BACK (Některá cívka brzdí za středem!)" if has_suckback else "✅ Čistý průlet bez suck-backu"

        txt2 = (
            f"VÝSTUPNÍ RYCHLOST: {sr['v_exit']:.2f} m/s ({sr['v_exit']*3.6:.1f} km/h) | Špičková rychlost v hlavni: {sr['v_max']:.2f} m/s\n"
            f"KINETICKÁ ENERGIE: {ek_mJ:.1f} mJ | Celková hmotnost: {self.p2['m_celkem_g']:.1f} g | Délka hlavně: {sr['delka_hlavne_mm']:.0f} mm | {suckback_txt}"
        )
        self.status_banner2.config(text=txt2, fg=(RED_WARN if has_suckback else PAPAYA))

        self.ax_vel.clear()
        self._styluj_graf(self.ax_vel)
        self.ax_vel.plot(sr["t_arr"], sr["v_arr"], color=PAPAYA, lw=2.5, label="Rychlost vozíčku v(t) [m/s]")
        self.ax_vel.set_title("Průběh rychlosti vozíčku v čase t [ms]", fontsize=9, fontweight="bold")
        self.ax_vel.set_xlabel("Čas od výstřelu t [ms]")
        self.ax_vel.set_ylabel("Rychlost v [m/s]")
        self.ax_vel.grid(True, color=GRID_COL, alpha=0.5)
        self.ax_vel.legend(loc="upper left", fontsize=8, facecolor=BG_DARK, edgecolor=GRID_COL, labelcolor=FG_TEXT)

        self.ax_energy.clear()
        self._styluj_graf(self.ax_energy)
        n_fazi = len(sr["energie_fazi_mJ"])
        x_idx = np.arange(1, n_fazi + 1)
        barvy = [PAPAYA if (e >= 0 and sb <= 0.5) else RED_WARN for e, sb in zip(sr["energie_fazi_mJ"], sr["suckback_fazi_mJ"])]
        bars = self.ax_energy.bar(x_idx, sr["energie_fazi_mJ"], color=barvy, edgecolor="#000000", width=0.55)
        self.ax_energy.axhline(0, color=FG_MUTED, lw=0.8)
        self.ax_energy.set_xticks(x_idx)
        self.ax_energy.set_xticklabels([f"Cívka #{i}" for i in x_idx], color=FG_TEXT)
        self.ax_energy.set_title("Předaná mechanická energie jednotlivými stupni ΔE [mJ] (Oranžová = Čistý tah, Červená = Suck-back!)", fontsize=9, fontweight="bold")
        self.ax_energy.set_xlabel("Stupeň katapultu")
        self.ax_energy.set_ylabel("Práce ΔE [mJ]")
        self.ax_energy.grid(True, axis="y", color=GRID_COL, alpha=0.5)

        max_bar = max(1.0, np.max(np.abs(sr["energie_fazi_mJ"])))
        self.ax_energy.set_ylim(min(0.0, np.min(sr["energie_fazi_mJ"]) * 1.25), max_bar * 1.25)

        for bar, val in zip(bars, sr["energie_fazi_mJ"]):
            offset = 0.04 * max_bar if val >= 0 else -0.12 * max_bar
            self.ax_energy.text(bar.get_x() + bar.get_width()/2.0, val + offset, f"{val:.1f} mJ", ha="center", color=FG_TEXT, fontsize=8, fontweight="bold")

        self.anim_frame_idx = len(sr["t_arr"]) - 1
        self._vykresli_snimek_animace(self.anim_frame_idx)

    def _vykresli_snimek_animace(self, idx):
        sr = self.sim_res
        idx = max(0, min(len(sr["t_arr"]) - 1, int(idx)))
        t_now = sr["t_arr"][idx]
        z_now = sr["z_arr"][idx]
        v_now = sr["v_arr"][idx]

        self.ax_anim.clear()
        self._styluj_graf(self.ax_anim)
        max_z = sr["delka_hlavne_mm"]
        self.ax_anim.set_xlim(-5, max_z + 10)
        self.ax_anim.set_ylim(-24, 24)

        half_g = self.p1["mezera"] / 2.0
        self.ax_anim.axhline(half_g, color=FG_MUTED, lw=2)
        self.ax_anim.axhline(-half_g, color=FG_MUTED, lw=2)

        r_coil = self.p1["d_vnejsi"] / 2.0
        w_coil = self.p1["sirka"]
        for i, z_c in enumerate(sr["stredy_civek_mm"]):
            stav = sr["led_state"][idx, i]
            coil_col = GREEN_OK if stav == 1 else (RED_WARN if stav == 2 else "#3B3F4C")
            self.ax_anim.add_patch(Rectangle((z_c - r_coil, half_g), 2*r_coil, w_coil + 3, facecolor=coil_col, edgecolor="#000000", alpha=0.85))
            self.ax_anim.add_patch(Rectangle((z_c - r_coil, -half_g - w_coil - 3), 2*r_coil, w_coil + 3, facecolor=coil_col, edgecolor="#000000", alpha=0.85))
            self.ax_anim.axvline(z_c, color=GRID_COL, linestyle=":", lw=0.9)

            led_col = "#00FF66" if stav == 1 else ("#FF2200" if stav == 2 else "#2C303C")
            self.ax_anim.add_patch(Circle((z_c, half_g + w_coil + 8), radius=2.8, facecolor=led_col, edgecolor="#000000"))
            self.ax_anim.text(z_c, -half_g - w_coil - 10, f"#{i+1}", ha="center", color=FG_TEXT, fontsize=8, fontweight="bold")

        mag_l = self.p1["mag_z"]
        mag_d = min(self.p1["mag_x"], self.p1["mezera"] - 1.0)
        self.ax_anim.add_patch(Rectangle((z_now - mag_l/2.0, -mag_d/2.0), mag_l, mag_d, facecolor=PAPAYA, edgecolor="#FFFFFF", lw=1.8))

        self.ax_anim.set_title(
            f"🎥 High-Speed Kamera: t = {t_now:.2f} ms | Poloha Z = {z_now:.1f} mm | Rychlost = {v_now:.2f} m/s  (🟢=Tah ON, 🔴=Flyback doběh)",
            fontsize=9, fontweight="bold"
        )
        self.ax_anim.set_xlabel("Poloha v hlavni Z [mm]")
        self.ax_anim.set_yticks([])
        self.canvas2.draw_idle()

    def _vykresli_tab3(self):
        ar = self.aero_res
        rezim_txt = "💥 PŘETAŽENÍ (STALL / LOOP OF DEATH) - Snižte úhel hlavně nebo upravte těžiště!" if ar["stalled"] else "✅ ČISTÝ KLOUZAVÝ LET (ZOOM-CLIMB)"
        banner_col = RED_WARN if ar["stalled"] else PAPAYA

        txt3 = (
            f"DOLET VLAŠTOVKY: {ar['dolet_m']:.2f} m | Max výška: {ar['peak_y']:.2f} m | Rychlost výstřelu v_0: {ar['v0']:.2f} m/s\n"
            f"AERODYNAMIKA: Klouzavost L/D = {ar['ld_ratio']:.2f} | Rovnovážná rychlost v_trim = {ar['v_trim_ms']:.2f} m/s | Úhel hlavně θ_gun = {ar['uhel_deg']:.1f}°\n"
            f"STAV LETU: {rezim_txt}"
        )
        self.status_banner3.config(text=txt3, fg=banner_col)

        self.ax_traj.clear()
        self._styluj_graf(self.ax_traj)
        traj_col = RED_WARN if ar["stalled"] else PAPAYA
        self.ax_traj.plot(ar["x_tr"], ar["y_tr"], color=traj_col, lw=2.8, label=f"Trajektorie letu (θ_gun = {ar['uhel_deg']:.1f}°, Dolet = {ar['dolet_m']:.2f} m)")
        self.ax_traj.fill_between(ar["x_tr"], 0, ar["y_tr"], color=traj_col, alpha=0.15)
        self.ax_traj.axhline(0, color=FG_MUTED, lw=1.2)
        self.ax_traj.scatter([0], [self.p3["h_stolu_m"]], color=CYAN_ACCENT, s=65, zorder=5, label=f"Ústí katapultu (výška {self.p3['h_stolu_m']:.2f} m)")
        self.ax_traj.set_title("2D Boční profil letu vlaštovky (Výška Y [m] vs. Vodorovná vzdálenost X [m])", fontsize=10, fontweight="bold")
        self.ax_traj.set_xlabel("Vodorovná vzdálenost od ústí katapultu X [m]")
        self.ax_traj.set_ylabel("Výška nad podlahou Y [m]")
        self.ax_traj.set_xlim(0, max(10.0, ar["dolet_m"] * 1.15))
        self.ax_traj.set_ylim(0, max(2.2, ar["peak_y"] * 1.2))
        self.ax_traj.legend(loc="upper right", fontsize=9, facecolor=BG_DARK, edgecolor=GRID_COL, labelcolor=FG_TEXT)
        self.ax_traj.grid(True, color=GRID_COL, alpha=0.5)

        self.canvas3.draw()

    # ---------------------------------------------------------------------
    # OVLÁDÁNÍ ANIMACE & ADMIN AUTO-OPTIMALIZACE
    # ---------------------------------------------------------------------
    def on_scrub_move(self, val):
        self.anim_running = False
        t_target = float(val)
        idx = int(np.searchsorted(self.sim_res["t_arr"], t_target))
        self._vykresli_snimek_animace(idx)

    def toggle_animace(self):
        if self.anim_running:
            self.anim_running = False
        else:
            self.anim_running = True
            self.anim_frame_idx = 0
            self._krok_animace()

    def _krok_animace(self):
        if not self.anim_running:
            return
        self._vykresli_snimek_animace(self.anim_frame_idx)
        if self.anim_frame_idx < len(self.sim_res["t_arr"]):
            self.scrub_var.set(self.sim_res["t_arr"][self.anim_frame_idx])
        self.anim_frame_idx += 3
        if self.anim_frame_idx < len(self.sim_res["t_arr"]):
            self.root.after(25, self._krok_animace)
        else:
            self.anim_running = False

    def admin_najdi_optimum(self):
        dprint("ADMIN: Automatické hledání optimálních časů sepnutí...")
        self._nacti_vstupy()
        n_fazi = self.p1["pocet_fazi"]
        stredy = [25.0 + i * self.p2["roztec_mm"] for i in range(n_fazi)]
        dt_ms = 0.05
        z_curr, v_curr = 13.0, 0.0
        m_tot_kg = self.p2["m_celkem_g"] / 1000.0
        f_treni = self.p2["mu_treni"] * m_tot_kg * 9.81

        nalezeny_on = [0.0] * n_fazi
        nalezeny_off = [5.0] * n_fazi
        byl_on = [False] * n_fazi
        byl_off = [False] * n_fazi

        for k in range(4000):
            t_ms = k * dt_ms
            f_mag = 0.0
            for i, z_c in enumerate(stredy):
                rel_z = z_curr - z_c
                if -15.0 <= rel_z <= -1.5:
                    if not byl_on[i]:
                        nalezeny_on[i] = round(t_ms, 1)
                        byl_on[i] = True
                    f_mag += np.interp(rel_z, self.faze_res["zs_osa_mm"], self.faze_res["Fz_osa"]) * 0.85
                elif rel_z > -1.5 and byl_on[i] and not byl_off[i]:
                    nalezeny_off[i] = round(t_ms, 1)
                    byl_off[i] = True

            a = max(0.0, f_mag - f_treni) / m_tot_kg if v_curr < 0.01 else (f_mag - f_treni) / m_tot_kg
            v_curr = max(0.0, v_curr + a * (dt_ms / 1000.0))
            z_curr += v_curr * dt_ms

        for i in range(n_fazi):
            self.casovani_vars[i][0].set(nalezeny_on[i])
            self.casovani_vars[i][1].set(max(1.0, round(nalezeny_off[i] - nalezeny_on[i], 1)))

        self.aktualizuj_vse()

if __name__ == "__main__":
    dprint("Spouštím Expedice Mars Telemetry & Coilgun Suite v5.0...")
    root = tk.Tk()
    app = ExpediceMarsSimV5(root)
    root.mainloop()