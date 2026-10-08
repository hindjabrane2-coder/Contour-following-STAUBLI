"""
Synchronisation d'axes -- coordination robot + axe externe.
Profils trapezoidal et en S, raccordements (zone fine), engrenage
electronique, profil de came, compensation de jeu.
TP robotique industrielle -- Staubli TX2-90.
"""
import numpy as np
import matplotlib.pyplot as plt
import csv


# --- Profils de vitesse ---

def profil_trapeze(distance, v_max, a_max, dt=0.001):
    """Profil de vitesse trapezoidal classique."""
    if distance < 1e-9:
        return np.array([0]), np.array([0]), np.array([0]), np.array([0])
    t_acc = v_max / a_max
    d_acc = 0.5 * a_max * t_acc**2
    if 2 * d_acc >= distance:
        t_acc = np.sqrt(distance / a_max)
        v_pic = a_max * t_acc
        t_cruise = 0.0
    else:
        v_pic = v_max
        t_cruise = (distance - 2 * d_acc) / v_max
    t_total = 2 * t_acc + t_cruise
    t = np.arange(0, t_total + dt, dt)
    pos, vel, acc = np.zeros_like(t), np.zeros_like(t), np.zeros_like(t)
    for i, ti in enumerate(t):
        if ti <= t_acc:
            acc[i] = a_max; vel[i] = a_max * ti; pos[i] = 0.5 * a_max * ti**2
        elif ti <= t_acc + t_cruise:
            vel[i] = v_pic; pos[i] = d_acc + v_pic * (ti - t_acc)
        else:
            td = ti - t_acc - t_cruise
            acc[i] = -a_max
            vel[i] = max(v_pic - a_max * td, 0)
            pos[i] = d_acc + v_pic * t_cruise + v_pic * td - 0.5 * a_max * td**2
    pos = np.minimum(pos, distance)
    return t, pos, np.maximum(vel, 0), acc


def profil_s_curve(distance, v_max, a_max, j_max, dt=0.001):
    """
    Profil en S (limitation du jerk).
    Le jerk est la derivee de l'acceleration ; le limiter reduit
    les vibrations mecaniques dans les reducteurs.
    """
    if distance < 1e-9:
        return np.array([0]), np.array([0]), np.array([0]), np.array([0])
    t_j = a_max / j_max
    t_acc = v_max / a_max
    d_acc = 0.5 * a_max * t_acc**2
    if 2 * d_acc >= distance:
        t_acc = np.sqrt(distance / a_max)
        v_pic = a_max * t_acc
        t_cruise = 0.0
    else:
        v_pic = v_max
        t_cruise = (distance - 2 * d_acc) / v_max
    t_total = 2 * t_acc + t_cruise
    t = np.arange(0, t_total + dt, dt)
    n = len(t)
    acc = np.zeros(n)
    # transitions lisses aux changements d'acceleration
    for i, ti in enumerate(t):
        if ti < t_j:
            acc[i] = j_max * ti
        elif ti < t_acc - t_j:
            acc[i] = a_max
        elif ti < t_acc:
            acc[i] = a_max - j_max * (ti - (t_acc - t_j))
        elif ti < t_acc + t_cruise:
            acc[i] = 0.0
        elif ti < t_acc + t_cruise + t_j:
            acc[i] = -j_max * (ti - t_acc - t_cruise)
        elif ti < t_total - t_j:
            acc[i] = -a_max
        else:
            acc[i] = -a_max + j_max * (ti - (t_total - t_j))
    vel, pos = np.zeros(n), np.zeros(n)
    for i in range(1, n):
        vel[i] = max(vel[i-1] + acc[i] * dt, 0)
        pos[i] = pos[i-1] + vel[i] * dt
    if pos[-1] > 0:
        pos = pos * distance / pos[-1]
    return t, pos, vel, acc


# --- Synchronisation multi-axes ---

def calculer_duree_axe(distance, v_max, a_max):
    if distance < 1e-9:
        return 0.0
    t_acc = v_max / a_max
    d_acc = 0.5 * a_max * t_acc**2
    if 2 * d_acc >= distance:
        return 2 * np.sqrt(distance / a_max)
    return 2 * t_acc + (distance - 2 * d_acc) / v_max

def synchroniser_axes(axes_config, profil='trapeze', dt=0.001):
    """Synchronise les axes pour qu'ils terminent en meme temps."""
    durees = [calculer_duree_axe(ax['distance'], ax['v_max'], ax['a_max'])
              for ax in axes_config]
    t_sync = max(durees)
    axe_limitant = np.argmax(durees)
    resultats = []
    for idx, ax in enumerate(axes_config):
        d = ax['distance']
        if d < 1e-9:
            resultats.append({'nom': ax['nom'], 't': np.array([0, t_sync]),
                              'pos': np.array([0, 0]), 'vel': np.array([0, 0]),
                              'acc': np.array([0, 0]), 'v_sync': 0, 'limitant': False})
            continue
        a = ax['a_max']
        discriminant = t_sync**2 - 4 * d / a
        v_sync = (t_sync - np.sqrt(discriminant)) * a / 2 if discriminant > 0 else np.sqrt(d * a)
        v_sync = min(v_sync, ax['v_max'])
        if profil == 'trapeze':
            t, pos, vel, acc = profil_trapeze(d, v_sync, a, dt)
        else:
            t, pos, vel, acc = profil_s_curve(d, v_sync, a, ax.get('j_max', a * 10), dt)
        resultats.append({'nom': ax['nom'], 't': t, 'pos': pos, 'vel': vel,
                          'acc': acc, 'v_sync': v_sync, 'limitant': (idx == axe_limitant)})
    return resultats, t_sync, axe_limitant


# --- Raccordement entre segments (zone fine) ---

def raccorder_segments(segments, zone_rayon=20.0, dt=0.001):
    """
    Enchaine des segments avec raccordement en zone fine
    (le robot ne s'arrete pas entre deux segments).
    """
    traj_x, traj_y, vit = [], [], []
    for seg_idx, seg in enumerate(segments):
        dx = seg['arrivee'][0] - seg['depart'][0]
        dy = seg['arrivee'][1] - seg['depart'][1]
        dist = np.sqrt(dx**2 + dy**2)
        n_pts = max(int(dist / (seg['v_max'] * dt)), 10)
        for si in np.linspace(0, 1, n_pts):
            x = seg['depart'][0] + si * dx
            y = seg['depart'][1] + si * dy
            if seg_idx < len(segments) - 1 and si > 0.9:
                prox = (si - 0.9) / 0.1
                seg_s = segments[seg_idx + 1]
                dx2 = seg_s['arrivee'][0] - seg_s['depart'][0]
                dy2 = seg_s['arrivee'][1] - seg_s['depart'][1]
                dist_coin = np.sqrt((x - seg['arrivee'][0])**2 + (y - seg['arrivee'][1])**2)
                if dist_coin < zone_rayon:
                    alpha = 1 - dist_coin / zone_rayon
                    x_s = seg_s['depart'][0] + prox * 0.1 * dx2
                    y_s = seg_s['depart'][1] + prox * 0.1 * dy2
                    x = x * (1 - alpha * 0.3) + x_s * alpha * 0.3
                    y = y * (1 - alpha * 0.3) + y_s * alpha * 0.3
            traj_x.append(x); traj_y.append(y); vit.append(seg['v_max'])
    return np.array(traj_x), np.array(traj_y), np.array(vit)


# --- Engrenage electronique et came ---

def engrenage_electronique(pos_maitre, ratio, offset=0.0):
    """Position esclave = ratio * maitre + offset."""
    return ratio * pos_maitre + offset

def profil_came(pos_maitre, n_lobes=3, amplitude=20.0, offset_came=0.0):
    """Relation non-lineaire maitre/esclave (mecanisme a came)."""
    pos_max = np.max(pos_maitre) if np.max(pos_maitre) > 0 else 1.0
    theta = pos_maitre / pos_max * 2 * np.pi * n_lobes
    return amplitude * (1 - np.cos(theta)) / 2 + offset_came


# --- Compensation de jeu (backlash) ---

def appliquer_backlash(commande, jeu=0.5):
    """Simule l'effet du jeu mecanique dans un reducteur."""
    n = len(commande)
    sortie = np.zeros(n)
    sortie[0] = commande[0]
    zone_morte = 0.0
    for i in range(1, n):
        zone_morte += commande[i] - commande[i-1]
        if zone_morte > jeu / 2:
            sortie[i] = sortie[i-1] + (zone_morte - jeu / 2)
            zone_morte = jeu / 2
        elif zone_morte < -jeu / 2:
            sortie[i] = sortie[i-1] + (zone_morte + jeu / 2)
            zone_morte = -jeu / 2
        else:
            sortie[i] = sortie[i-1]
    return sortie

def compenser_backlash(commande, jeu=0.5):
    """Compensation du jeu par pre-charge de la commande."""
    n = len(commande)
    compensee = np.zeros(n)
    compensee[0] = commande[0]
    for i in range(1, n):
        delta = commande[i] - commande[i-1]
        if delta > 0:
            compensee[i] = commande[i] + jeu / 2
        elif delta < 0:
            compensee[i] = commande[i] - jeu / 2
        else:
            compensee[i] = compensee[i-1]
    return compensee


# --- Override de vitesse ---

def appliquer_override(vel, override_pct, t, t_debut, t_fin):
    """Changement de vitesse en cours de mouvement (pourcentage de la nominale)."""
    vel_mod = vel.copy()
    mask = (t >= t_debut) & (t <= t_fin)
    vel_mod[mask] = vel[mask] * override_pct / 100.0
    return vel_mod


# --- Export ---

def exporter_profils(chemin, resultats, t_sync, dt=0.001):
    t_commun = np.arange(0, t_sync + dt, dt)
    with open(chemin, 'w', newline='') as f:
        writer = csv.writer(f, delimiter=';')
        entete = ['temps_s']
        for res in resultats:
            entete.extend([f'{res["nom"]}_pos', f'{res["nom"]}_vel'])
        writer.writerow(entete)
        for ti in t_commun:
            ligne = [f'{ti:.4f}']
            for res in resultats:
                ligne.append(f'{np.interp(ti, res["t"], res["pos"]):.3f}')
                ligne.append(f'{np.interp(ti, res["t"], res["vel"]):.3f}')
            writer.writerow(ligne)


# --- Programme principal ---

if __name__ == '__main__':
    config = [
        {'nom': 'Axe 1 (base)',   'distance': 500, 'v_max': 800, 'a_max': 3000, 'j_max': 30000},
        {'nom': 'Axe 2 (epaule)', 'distance': 300, 'v_max': 600, 'a_max': 2500, 'j_max': 25000},
        {'nom': 'Axe 3 (coude)',  'distance': 200, 'v_max': 700, 'a_max': 3000, 'j_max': 30000},
        {'nom': 'Axe ext (rail)', 'distance': 800, 'v_max': 500, 'a_max': 1500, 'j_max': 15000},
    ]

    print("=== Comparaison profil trapeze vs S-curve ===\n")
    for ptype in ['trapeze', 's_curve']:
        fn = profil_trapeze if ptype == 'trapeze' else profil_s_curve
        args = (500, 800, 3000) if ptype == 'trapeze' else (500, 800, 3000, 30000)
        t, pos, vel, acc = fn(*args)
        jerk = np.gradient(acc, t) if len(t) > 1 else np.array([0])
        print(f"  {ptype:10s} : duree={t[-1]:.4f} s, jerk max={np.max(np.abs(jerk)):.0f} mm/s^3")

    print("\n=== Synchronisation multi-axes ===\n")
    for prof in ['trapeze', 's_curve']:
        res, t_sync, idx_lim = synchroniser_axes(config, profil=prof)
        print(f"  Profil {prof:10s} : T_sync={t_sync:.4f} s, limitant={config[idx_lim]['nom']}")
        for i, r in enumerate(res):
            v = r.get('v_sync', 0)
            pct = v / config[i]['v_max'] * 100 if config[i]['v_max'] > 0 else 0
            lim = ' [LIM]' if r['limitant'] else ''
            print(f"    {r['nom']:22s} : v_sync={v:6.0f} mm/s ({pct:4.0f}%){lim}")

    print("\n=== Raccordement zone fine ===")
    segments = [
        {'depart': [0, 0], 'arrivee': [100, 0], 'v_max': 500},
        {'depart': [100, 0], 'arrivee': [100, 80], 'v_max': 400},
        {'depart': [100, 80], 'arrivee': [200, 80], 'v_max': 500},
        {'depart': [200, 80], 'arrivee': [200, 0], 'v_max': 400},
    ]
    traj_x, traj_y, _ = raccorder_segments(segments, zone_rayon=15.0)
    print(f"  {len(segments)} segments, {len(traj_x)} points")

    print("\n=== Engrenage et came ===")
    res_sync, t_sync, _ = synchroniser_axes(config, profil='trapeze')
    pos_m = res_sync[0]['pos']
    pos_gear = engrenage_electronique(pos_m, ratio=1.5)
    pos_came = profil_came(pos_m, n_lobes=4, amplitude=15.0)
    print(f"  Engrenage r=1.5 : finale={pos_gear[-1]:.1f} mm")
    print(f"  Came 4 lobes    : amplitude=15 mm")

    print("\n=== Compensation backlash ===")
    jeu = 0.8
    cmd = res_sync[1]['pos']
    sortie_sc = appliquer_backlash(cmd, jeu)
    sortie_ac = appliquer_backlash(compenser_backlash(cmd, jeu), jeu)
    print(f"  Jeu={jeu} mm, err max sans comp={np.max(np.abs(cmd[:len(sortie_sc)] - sortie_sc)):.3f} mm, "
          f"avec comp={np.max(np.abs(cmd[:len(sortie_ac)] - sortie_ac)):.3f} mm")

    exporter_profils('profils_synchronises.csv', res_sync, t_sync)

    # --- Visualisation ---
    fig, axes_fig = plt.subplots(2, 3, figsize=(16, 9))
    couleurs = ['steelblue', 'coral', 'green', 'purple']

    ax = axes_fig[0, 0]
    for i, r in enumerate(res_sync):
        d = config[i]['distance']
        if d > 0: ax.plot(r['t'], r['pos'] / d * 100, color=couleurs[i], linewidth=2, label=r['nom'])
    ax.set_xlabel('Temps (s)'); ax.set_ylabel('Position (%)')
    ax.set_title('Positions normalisees'); ax.legend(fontsize=7); ax.grid(True, alpha=0.3)

    ax = axes_fig[0, 1]
    for i, r in enumerate(res_sync):
        if config[i]['distance'] > 0:
            ax.plot(r['t'], r['vel'], color=couleurs[i], linewidth=2, label=r['nom'])
    ax.set_xlabel('Temps (s)'); ax.set_ylabel('Vitesse (mm/s)')
    ax.set_title('Profils de vitesse'); ax.legend(fontsize=7); ax.grid(True, alpha=0.3)

    ax = axes_fig[0, 2]
    t_tr, _, _, acc_tr = profil_trapeze(500, 800, 3000)
    t_sc, _, _, acc_sc = profil_s_curve(500, 800, 3000, 30000)
    ax.plot(t_tr, acc_tr, 'b-', linewidth=1.5, label='Trapeze')
    ax.plot(t_sc, acc_sc, 'r-', linewidth=1.5, label='S-curve')
    ax.set_xlabel('Temps (s)'); ax.set_ylabel('Acc. (mm/s^2)')
    ax.set_title('Trapeze vs S-curve'); ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

    ax = axes_fig[1, 0]
    ax.plot(traj_x, traj_y, 'b-', linewidth=2)
    for seg in segments:
        ax.plot(*seg['depart'], 'go', markersize=6)
        ax.plot(*seg['arrivee'], 'rs', markersize=6)
    ax.set_xlabel('X (mm)'); ax.set_ylabel('Y (mm)')
    ax.set_title('Raccordement zone fine'); ax.set_aspect('equal'); ax.grid(True, alpha=0.3)

    ax = axes_fig[1, 1]
    ax.plot(pos_m, pos_gear, 'b-', linewidth=1.5, label='Engrenage (r=1.5)')
    ax.plot(pos_m, pos_came, 'r-', linewidth=1.5, label='Came (4 lobes)')
    ax.set_xlabel('Maitre (mm)'); ax.set_ylabel('Esclave (mm)')
    ax.set_title('Engrenage et came'); ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

    ax = axes_fig[1, 2]
    t_ax2 = res_sync[1]['t']
    ax.plot(t_ax2, cmd, 'k-', linewidth=1, label='Consigne')
    ax.plot(t_ax2, sortie_sc, 'r--', linewidth=1, label='Sans comp.')
    ax.plot(t_ax2, sortie_ac, 'g-', linewidth=1, alpha=0.7, label='Avec comp.')
    ax.set_xlabel('Temps (s)'); ax.set_ylabel('Position (mm)')
    ax.set_title(f'Backlash (jeu={jeu} mm)'); ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

    plt.suptitle("Synchronisation d'axes -- Staubli TX2-90 + axe externe",
                  fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.savefig('axis_sync.png', dpi=150)
    print("\nFigure sauvegardee : axis_sync.png")
