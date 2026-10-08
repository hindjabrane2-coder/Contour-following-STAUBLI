"""
Regulation PID en effort.
Plusieurs types de correcteurs, auto-reglage Ziegler-Nichols, analyse
frequentielle, predicteur de Smith, comparaison de performances.
TP robotique industrielle -- Staubli TX2-90.
"""
import numpy as np
import matplotlib.pyplot as plt


# --- Correcteurs ---

class CorrecteurP:
    """Correcteur proportionnel simple."""
    def __init__(self, kp, u_min=-50, u_max=50):
        self.kp, self.u_min, self.u_max = kp, u_min, u_max
    def calculer(self, consigne, mesure):
        return np.clip(self.kp * (consigne - mesure), self.u_min, self.u_max)
    def reset(self):
        pass


class CorrecteurPI:
    """Correcteur PI avec anti-windup."""
    def __init__(self, kp, ki, dt, i_max=50.0, u_min=-50, u_max=50):
        self.kp, self.ki, self.dt = kp, ki, dt
        self.i_max, self.u_min, self.u_max = i_max, u_min, u_max
        self.integral = 0.0
    def calculer(self, consigne, mesure):
        erreur = consigne - mesure
        self.integral = np.clip(self.integral + erreur * self.dt, -self.i_max, self.i_max)
        return np.clip(self.kp * erreur + self.ki * self.integral, self.u_min, self.u_max)
    def reset(self):
        self.integral = 0.0


class CorrecteurPID:
    """PID complet avec anti-windup, filtrage derivee, option derivee sur mesure."""
    def __init__(self, kp, ki, kd, dt, i_max=50.0, d_filter=0.1,
                 u_min=-50, u_max=50, derivee_sur_mesure=False):
        self.kp, self.ki, self.kd, self.dt = kp, ki, kd, dt
        self.i_max, self.d_filter = i_max, d_filter
        self.u_min, self.u_max = u_min, u_max
        self.derivee_sur_mesure = derivee_sur_mesure
        self.integral = 0.0
        self.erreur_prec = 0.0
        self.mesure_prec = 0.0
        self.derivee_filtree = 0.0

    def calculer(self, consigne, mesure):
        erreur = consigne - mesure
        p = self.kp * erreur
        self.integral = np.clip(self.integral + erreur * self.dt, -self.i_max, self.i_max)
        i = self.ki * self.integral
        if self.derivee_sur_mesure:
            derivee_brute = -(mesure - self.mesure_prec) / self.dt
        else:
            derivee_brute = (erreur - self.erreur_prec) / self.dt
        self.derivee_filtree = (self.d_filter * derivee_brute
                                 + (1 - self.d_filter) * self.derivee_filtree)
        d = self.kd * self.derivee_filtree
        self.erreur_prec = erreur
        self.mesure_prec = mesure
        return np.clip(p + i + d, self.u_min, self.u_max)

    def reset(self):
        self.integral = self.erreur_prec = self.mesure_prec = self.derivee_filtree = 0.0


class CorrecteurPID_FF(CorrecteurPID):
    """PID avec feedforward sur la variation de consigne."""
    def __init__(self, kp, ki, kd, kff, dt, **kwargs):
        super().__init__(kp, ki, kd, dt, **kwargs)
        self.kff, self.consigne_prec = kff, 0.0
    def calculer(self, consigne, mesure):
        u_pid = super().calculer(consigne, mesure)
        u_ff = self.kff * (consigne - self.consigne_prec) / self.dt
        self.consigne_prec = consigne
        return np.clip(u_pid + u_ff, self.u_min, self.u_max)


class PredicteurSmith:
    """Predicteur de Smith : compense le retard pur dans la boucle."""
    def __init__(self, correcteur, raideur_modele, retard_echantillons):
        self.correcteur = correcteur
        self.raideur = raideur_modele
        self.retard = retard_echantillons
        self.buffer = [0.0] * (retard_echantillons + 1)
        self.pos_modele = 0.0

    def calculer(self, consigne, mesure):
        sortie_actuelle = self.raideur * self.pos_modele
        sortie_retardee = self.raideur * self.buffer[0]
        mesure_corrigee = mesure - sortie_retardee + sortie_actuelle
        u = self.correcteur.calculer(consigne, mesure_corrigee)
        self.pos_modele += u * self.correcteur.dt
        self.buffer.append(self.pos_modele)
        self.buffer.pop(0)
        return u

    def reset(self):
        self.correcteur.reset()
        self.buffer = [0.0] * (self.retard + 1)
        self.pos_modele = 0.0


# --- Auto-reglage Ziegler-Nichols ---

def ziegler_nichols(raideur, dt, duree=3.0):
    """
    Methode de Ziegler-Nichols en boucle fermee.
    On cherche le gain critique Ku et la periode Tu.
    """
    n = int(duree / dt)
    ku_estime, tu_estime = 0, 0
    for kp_test in np.arange(0.1, 20.0, 0.1):
        t = np.arange(n) * dt
        force, pos = np.zeros(n), np.zeros(n)
        pos[0] = 15.0 / raideur
        for i in range(1, n):
            force[i-1] = raideur * pos[i-1] + np.random.normal(0, 0.1)
            pos[i] = max(pos[i-1] + kp_test * (15.0 - force[i-1]) * dt, 0)
        force[-1] = raideur * pos[-1]
        # detecter les oscillations dans la seconde moitie
        moitie = n // 2
        erreur_2 = force[moitie:] - 15.0
        pics = [i for i in range(1, len(erreur_2) - 1)
                if erreur_2[i] > erreur_2[i-1] and erreur_2[i] > erreur_2[i+1]]
        if len(pics) >= 4:
            amplitudes = [abs(erreur_2[p]) for p in pics]
            if amplitudes[-1] > amplitudes[0] * 0.8:
                ku_estime = kp_test
                tu_estime = np.mean([(pics[j+1] - pics[j]) * dt for j in range(len(pics)-1)])
                break
    if ku_estime == 0:
        ku_estime, tu_estime = 5.0, 0.2
    return ku_estime, tu_estime, {
        'P':   {'kp': 0.50 * ku_estime, 'ki': 0, 'kd': 0},
        'PI':  {'kp': 0.45 * ku_estime, 'ki': 0.54 * ku_estime / tu_estime, 'kd': 0},
        'PID': {'kp': 0.60 * ku_estime, 'ki': 1.20 * ku_estime / tu_estime,
                'kd': 0.075 * ku_estime * tu_estime},
    }


# --- Simulation du procede ---

def simuler_contact(correcteur, f_consigne=15.0, duree=2.0, dt=0.001,
                     raideur_base=100.0, bruit_capteur=0.3, retard=0):
    """Simule le contact avec changement de raideur et retard optionnel."""
    n = int(duree / dt)
    t = np.arange(n) * dt
    raideur = np.ones(n) * raideur_base
    for i in range(n):
        if 0.8 < t[i] < 1.2:
            raideur[i] = raideur_base * 0.4
        raideur[i] *= (1 + 0.1 * np.sin(2 * np.pi * 3 * t[i]))
    position, force, commande = np.zeros(n), np.zeros(n), np.zeros(n)
    position[0] = f_consigne / raideur[0]
    buffer_retard = [f_consigne] * (retard + 1)
    for i in range(1, n):
        force[i-1] = raideur[i-1] * position[i-1] + np.random.normal(0, bruit_capteur)
        buffer_retard.append(force[i-1])
        commande[i] = correcteur.calculer(f_consigne, buffer_retard.pop(0))
        position[i] = max(position[i-1] + commande[i] * dt, 0)
    force[-1] = raideur[-1] * position[-1]
    return t, force, position, commande, raideur


# --- Analyse de la reponse ---

def analyser_reponse(t, force, f_consigne):
    erreur = force - f_consigne
    i_10 = next((i for i in range(len(force)) if force[i] >= 0.1 * f_consigne), None)
    i_90 = next((i for i in range(len(force)) if force[i] >= 0.9 * f_consigne), None)
    t_montee = (t[i_90] - t[i_10]) if i_10 is not None and i_90 is not None else None
    depassement = max(0, (np.max(force[:len(force)//2]) - f_consigne) / f_consigne * 100)
    t_etablissement = None
    for i in range(len(t) - 1, -1, -1):
        if abs(erreur[i]) > f_consigne * 0.05:
            t_etablissement = t[min(i + 1, len(t) - 1)]
            break
    err_stat = np.mean(erreur[int(0.8 * len(force)):])
    return {'t_montee': t_montee, 'depassement': depassement,
            't_etablissement': t_etablissement, 'err_statique': err_stat,
            'mae': np.mean(np.abs(erreur))}


# --- Reponse frequentielle ---

def reponse_frequentielle(correcteur_cls, args, raideur, freqs, dt=0.001, duree=2.0):
    """Reponse en frequence par balayage sinusoidal."""
    gains_db, phases_deg = [], []
    for freq in freqs:
        corr = correcteur_cls(**args, dt=dt)
        n = int(duree / dt)
        t = np.arange(n) * dt
        pos, force = np.zeros(n), np.zeros(n)
        pos[0] = 15.0 / raideur
        for i in range(1, n):
            force[i-1] = raideur * pos[i-1]
            cons = 15.0 + 5.0 * np.sin(2 * np.pi * freq * t[i])
            pos[i] = max(pos[i-1] + corr.calculer(cons, force[i-1]) * dt, 0)
        force[-1] = raideur * pos[-1]
        moitie = n // 2
        f_2 = force[moitie:] - 15.0
        c_2 = 5.0 * np.sin(2 * np.pi * freq * t[moitie:])
        amp = (np.max(f_2) - np.min(f_2)) / 2
        gains_db.append(20 * np.log10(max(amp / 5.0, 1e-10)))
        corr_xy = np.correlate(f_2 - np.mean(f_2), c_2 - np.mean(c_2), 'full')
        decalage = np.argmax(corr_xy) - len(f_2) + 1
        phase = ((-decalage * dt * freq * 360 + 180) % 360) - 180
        phases_deg.append(phase)
    return np.array(gains_db), np.array(phases_deg)


# --- Windup ---

def demo_windup(f_consigne=15.0, duree=3.0, dt=0.001, raideur=100.0):
    """Demonstre l'effet du windup et l'anti-windup."""
    resultats = {}
    for mode, i_max in [('sans anti-windup', 1e6), ('avec anti-windup', 10.0)]:
        pid = CorrecteurPID(kp=0.8, ki=3.0, kd=0.05, dt=dt, i_max=i_max)
        n = int(duree / dt)
        t = np.arange(n) * dt
        force, pos, integ = np.zeros(n), np.zeros(n), np.zeros(n)
        pos[0] = f_consigne / raideur
        for i in range(1, n):
            cons = f_consigne if t[i] < 1.0 else f_consigne * 0.5
            force[i-1] = raideur * pos[i-1]
            cmd = pid.calculer(cons, force[i-1])
            integ[i] = pid.integral
            pos[i] = max(pos[i-1] + cmd * dt, 0)
        force[-1] = raideur * pos[-1]
        resultats[mode] = {'t': t, 'force': force, 'integral': integ}
    return resultats


# --- Robustesse ---

def analyse_robustesse(raideur_nom=100.0, dt=0.001, n_essais=20):
    raideurs = np.linspace(30, 200, n_essais)
    resultats = []
    for k in raideurs:
        pid = CorrecteurPID(kp=0.8, ki=2.0, kd=0.05, dt=dt)
        t, force, _, _, _ = simuler_contact(pid, duree=1.5, dt=dt,
                                              raideur_base=k, bruit_capteur=0.1)
        resultats.append(analyser_reponse(t, force, 15.0))
    return raideurs, resultats


# --- Programme principal ---

if __name__ == '__main__':
    np.random.seed(42)
    dt = 0.001
    f_cible = 15.0
    raideur_nom = 100.0

    print("=== Auto-reglage Ziegler-Nichols ===\n")
    ku, tu, gains_zn = ziegler_nichols(raideur_nom, dt)
    print(f"  Ku={ku:.2f}, Tu={tu:.4f} s")
    for nom, g in gains_zn.items():
        print(f"  {nom:4s} : Kp={g['kp']:.3f}, Ki={g['ki']:.3f}, Kd={g['kd']:.4f}")

    print("\n=== Comparaison des correcteurs ===\n")
    correcteurs = {
        'P':        CorrecteurP(kp=1.5),
        'PI':       CorrecteurPI(kp=0.8, ki=2.0, dt=dt),
        'PID':      CorrecteurPID(kp=0.8, ki=2.0, kd=0.05, dt=dt),
        'PID (ZN)': CorrecteurPID(kp=gains_zn['PID']['kp'], ki=gains_zn['PID']['ki'],
                                    kd=gains_zn['PID']['kd'], dt=dt),
        'PID + FF': CorrecteurPID_FF(kp=0.8, ki=2.0, kd=0.05, kff=0.1, dt=dt),
    }
    resultats_comp = {}
    print(f"  {'Correcteur':15s} {'t_montee':>10s} {'Depas.%':>8s} {'t_etab':>8s} {'MAE':>7s}")
    print("  " + "-" * 50)
    for nom, corr in correcteurs.items():
        corr.reset(); np.random.seed(42)
        t, force, pos, cmd, raid = simuler_contact(corr, f_consigne=f_cible, dt=dt)
        m = analyser_reponse(t, force, f_cible)
        resultats_comp[nom] = {'t': t, 'force': force, 'metriques': m}
        tm = f"{m['t_montee']*1000:.0f} ms" if m['t_montee'] else "  --  "
        te = f"{m['t_etablissement']*1000:.0f} ms" if m['t_etablissement'] else "  --  "
        print(f"  {nom:15s} {tm:>10s} {m['depassement']:>7.1f}% {te:>8s} {m['mae']:>6.2f} N")

    print("\n=== Derivative kick ===")
    for sur_m in [False, True]:
        pid = CorrecteurPID(kp=0.8, ki=2.0, kd=0.15, dt=dt, derivee_sur_mesure=sur_m)
        np.random.seed(42)
        _, _, _, cmd, _ = simuler_contact(pid, f_consigne=f_cible, dt=dt)
        print(f"  {'mesure' if sur_m else 'erreur':8s} : pic cmd = {np.max(np.abs(cmd[:200])):.2f}")

    print("\n=== Predicteur de Smith (retard 20 ms) ===")
    retard = 20
    for mode in ['PID seul', 'Smith']:
        pid = CorrecteurPID(kp=0.8, ki=2.0, kd=0.05, dt=dt)
        corr = PredicteurSmith(pid, raideur_nom, retard) if mode == 'Smith' else pid
        np.random.seed(42)
        t, force, _, _, _ = simuler_contact(corr, f_consigne=f_cible, dt=dt, retard=retard)
        m = analyser_reponse(t, force, f_cible)
        print(f"  {mode:12s} : depas={m['depassement']:.1f}%, MAE={m['mae']:.2f} N")

    print("\n=== Windup integral ===")
    res_windup = demo_windup()
    for mode, data in res_windup.items():
        print(f"  {mode:22s} : pic integral = {np.max(np.abs(data['integral'])):.1f}")

    print("\n=== Robustesse ===")
    raideurs_test, res_robust = analyse_robustesse(raideur_nom, dt)
    deps = [r['depassement'] for r in res_robust]
    maes = [r['mae'] for r in res_robust]
    print(f"  Raideur [{raideurs_test[0]:.0f}, {raideurs_test[-1]:.0f}] : "
          f"depas=[{min(deps):.1f}, {max(deps):.1f}]%, MAE=[{min(maes):.2f}, {max(maes):.2f}] N")

    print("\n=== Reponse frequentielle ===")
    freqs = np.logspace(-0.5, 2.0, 30)
    gains_bode, phases_bode = reponse_frequentielle(
        CorrecteurPID, {'kp': 0.8, 'ki': 2.0, 'kd': 0.05}, raideur_nom, freqs, dt)
    idx_180 = np.argmin(np.abs(phases_bode + 180))
    idx_0db = np.argmin(np.abs(gains_bode))
    print(f"  Marge gain  = {-gains_bode[idx_180]:.1f} dB a {freqs[idx_180]:.1f} Hz")
    print(f"  Marge phase = {180 + phases_bode[idx_0db]:.1f} deg a {freqs[idx_0db]:.1f} Hz")

    # --- Visualisation 6 sous-graphiques ---
    fig, axes = plt.subplots(3, 2, figsize=(14, 12))
    couleurs = ['steelblue', 'coral', 'green', 'purple', 'orange']

    ax = axes[0, 0]
    for idx, (nom, d) in enumerate(resultats_comp.items()):
        ax.plot(d['t'], d['force'], color=couleurs[idx], linewidth=1, alpha=0.8, label=nom)
    ax.axhline(y=f_cible, color='k', linestyle='--', linewidth=0.8)
    ax.fill_between([0, 2], f_cible * 0.95, f_cible * 1.05, alpha=0.08, color='green')
    ax.set_xlabel('Temps (s)'); ax.set_ylabel('Force (N)')
    ax.set_title('Comparaison des correcteurs'); ax.legend(fontsize=7); ax.grid(True, alpha=0.3)

    ax = axes[0, 1]
    for idx, (nom, d) in enumerate(resultats_comp.items()):
        ax.plot(d['t'][:500], d['force'][:500], color=couleurs[idx], linewidth=1.2, label=nom)
    ax.axhline(y=f_cible, color='k', linestyle='--', linewidth=0.8)
    ax.set_xlabel('Temps (s)'); ax.set_ylabel('Force (N)')
    ax.set_title('Transitoire (500 ms)'); ax.legend(fontsize=7); ax.grid(True, alpha=0.3)

    ax = axes[1, 0]
    ax.semilogx(freqs, gains_bode, 'b-', linewidth=2)
    ax.axhline(y=0, color='k', linewidth=0.5)
    ax.axhline(y=-3, color='r', linestyle='--', linewidth=0.8, label='-3 dB')
    ax.set_xlabel('Frequence (Hz)'); ax.set_ylabel('Gain (dB)')
    ax.set_title('Bode -- Gain'); ax.legend(fontsize=8); ax.grid(True, alpha=0.3, which='both')

    ax = axes[1, 1]
    ax.semilogx(freqs, phases_bode, 'b-', linewidth=2)
    ax.axhline(y=-180, color='r', linestyle='--', linewidth=0.8)
    ax.set_xlabel('Frequence (Hz)'); ax.set_ylabel('Phase (deg)')
    ax.set_title('Bode -- Phase'); ax.grid(True, alpha=0.3, which='both')

    ax = axes[2, 0]
    for mode, d in res_windup.items():
        ax.plot(d['t'], d['force'], 'r-' if 'sans' in mode else 'g-', linewidth=1.2, label=mode)
    ax.axhline(y=f_cible, color='k', linestyle='--', linewidth=0.8)
    ax.axhline(y=f_cible * 0.5, color='gray', linestyle=':', label='consigne t>1s')
    ax.set_xlabel('Temps (s)'); ax.set_ylabel('Force (N)')
    ax.set_title('Windup integral'); ax.legend(fontsize=7); ax.grid(True, alpha=0.3)

    ax = axes[2, 1]
    ax.plot(raideurs_test, deps, 'bo-', linewidth=1.5, markersize=4, label='Depassement (%)')
    ax.axvline(x=raideur_nom, color='k', linestyle='--', linewidth=0.8)
    ax2 = ax.twinx()
    ax2.plot(raideurs_test, maes, 'rs-', linewidth=1.5, markersize=4, label='MAE (N)')
    ax2.set_ylabel('MAE (N)', color='r')
    ax.set_xlabel('Raideur (N/mm)'); ax.set_ylabel('Depassement (%)', color='b')
    ax.set_title('Robustesse'); ax.legend(loc='upper left', fontsize=7)
    ax2.legend(loc='upper right', fontsize=7); ax.grid(True, alpha=0.3)

    plt.suptitle('Regulation PID en effort -- Staubli TX2-90', fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.savefig('force_control.png', dpi=150)
    print("\nFigure sauvegardee : force_control.png")
