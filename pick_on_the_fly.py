"""
Prise a la volee (pick on the fly) sur convoyeur.
Gestion de multiples objets, suivi par encodeur, strategies d'approche,
cycle complet pick-and-place avec optimisation du temps de cycle.
TP robotique industrielle -- Staubli TX2-90.
"""
import numpy as np
import matplotlib.pyplot as plt
from collections import deque
from dataclasses import dataclass


# --- Parametres globaux ---

V_CONVOYEUR = 200       # mm/s
LARGEUR_CONVOYEUR = 400 # mm
Y_CONVOYEUR = 300       # mm (centre du convoyeur)
VMAX_ROBOT = 1000       # mm/s
AMAX_ROBOT = 5000       # mm/s^2
X_DETECTION = 0         # mm (capteur de detection)
X_PRISE_MIN = 200       # mm (debut zone de prise)
X_PRISE_MAX = 600       # mm (fin zone de prise)
Z_APPROCHE = 100        # mm (hauteur securite)
Z_PRISE = 5             # mm (hauteur de prise)
P_DEPOT = np.array([400, -200, 50])  # position du bac de depot
Z_DEPOT = 30
P_REPOS = np.array([100, 0, 250])    # position d'attente

TEMPS_FERMETURE = 0.05  # s (gripper)
TEMPS_OUVERTURE = 0.04  # s
TOLERANCE_PRISE = 3.0   # mm (ecart max acceptable)


# --- Modele d'encodeur convoyeur ---

@dataclass
class EncodeurConvoyeur:
    """Simulation de l'encodeur incremental du convoyeur."""
    resolution: float = 0.5    # mm par impulsion
    bruit_position: float = 0.2
    latence: float = 0.002     # s
    _position: float = 0.0

    def lire_position(self, position_reelle):
        pos_quant = np.round(position_reelle / self.resolution) * self.resolution
        self._position = pos_quant + np.random.normal(0, self.bruit_position)
        return self._position

    def predire_position(self, t_prediction, vitesse):
        return self._position + vitesse * (t_prediction + self.latence)


# --- Objet sur le convoyeur ---

@dataclass
class ObjetConvoyeur:
    """Un objet detecte sur le convoyeur."""
    id: int
    x_detection: float
    t_detection: float
    y_offset: float = 0.0
    largeur: float = 40.0
    hauteur: float = 25.0
    pris: bool = False
    rate: bool = False

    def position_x(self, t, vitesse=V_CONVOYEUR):
        return self.x_detection + vitesse * (t - self.t_detection)

    def dans_zone_prise(self, t, vitesse=V_CONVOYEUR):
        x = self.position_x(t, vitesse)
        return X_PRISE_MIN <= x <= X_PRISE_MAX

    def sorti_zone(self, t, vitesse=V_CONVOYEUR):
        return self.position_x(t, vitesse) > X_PRISE_MAX


# --- File d'objets ---

class FileObjets:
    """FIFO des objets detectes en attente de prise."""
    def __init__(self):
        self.file = deque()
        self.compteur = 0
        self.historique = []

    def ajouter(self, t, y_offset=0.0):
        self.compteur += 1
        obj = ObjetConvoyeur(id=self.compteur, x_detection=X_DETECTION,
                              t_detection=t, y_offset=y_offset)
        self.file.append(obj)
        self.historique.append(obj)
        return obj

    def prochain(self):
        while self.file and (self.file[0].pris or self.file[0].rate):
            self.file.popleft()
        return self.file[0] if self.file else None

    def marquer_pris(self, obj):
        obj.pris = True

    def marquer_rate(self, obj):
        obj.rate = True


# --- Probabilite de prise ---

def probabilite_prise(erreur_sync, tolerance=TOLERANCE_PRISE):
    """Modele gaussien de reussite en fonction de l'erreur de synchronisation."""
    sigma = tolerance / 2.0
    return 0.95 * np.exp(-0.5 * (erreur_sync / sigma)**2)


# --- Strategies d'approche ---

def approche_dessus(p_depart, p_cible, t_dispo, dt=0.001):
    """Approche par le dessus : positionnement, synchronisation, descente."""
    t = np.arange(0, t_dispo, dt)
    n = len(t)
    pos = np.zeros((n, 3))
    t1, t2, t3 = t_dispo * 0.35, t_dispo * 0.45, t_dispo * 0.20

    for i in range(n):
        ti = t[i]
        if ti < t1:
            frac = ti / t1
            pos[i, 0] = p_depart[0] + frac * (p_cible[0] - V_CONVOYEUR * t_dispo * 0.4 - p_depart[0])
            pos[i, 1] = p_depart[1] + frac * (Y_CONVOYEUR + p_cible[1] - p_depart[1])
            pos[i, 2] = p_depart[2] + frac * (Z_APPROCHE - p_depart[2])
        elif ti < t1 + t2:
            frac = (ti - t1) / t2
            x_conv = X_DETECTION + V_CONVOYEUR * ti
            pos[i, 0] = x_conv + (1 - frac) * 40
            pos[i, 1] = Y_CONVOYEUR
            pos[i, 2] = Z_APPROCHE + frac * (Z_PRISE + 15 - Z_APPROCHE)
        else:
            frac = (ti - t1 - t2) / t3
            pos[i, 0] = X_DETECTION + V_CONVOYEUR * ti
            pos[i, 1] = Y_CONVOYEUR
            pos[i, 2] = Z_PRISE + (1 - frac) * 15
    return t, pos

def approche_laterale(p_depart, p_cible, t_dispo, dt=0.001):
    """Approche laterale : le robot arrive par le cote du convoyeur."""
    t = np.arange(0, t_dispo, dt)
    n = len(t)
    pos = np.zeros((n, 3))
    t1, t2, t3 = t_dispo * 0.40, t_dispo * 0.35, t_dispo * 0.25
    y_cote = Y_CONVOYEUR - LARGEUR_CONVOYEUR / 2 - 50

    for i in range(n):
        ti = t[i]
        if ti < t1:
            frac = ti / t1
            pos[i, 0] = p_depart[0] + frac * (p_cible[0] - V_CONVOYEUR * t_dispo * 0.3 - p_depart[0])
            pos[i, 1] = p_depart[1] + frac * (y_cote - p_depart[1])
            pos[i, 2] = p_depart[2] + frac * (Z_PRISE + 30 - p_depart[2])
        elif ti < t1 + t2:
            frac = (ti - t1) / t2
            pos[i, 0] = X_DETECTION + V_CONVOYEUR * ti + (1 - frac) * 30
            pos[i, 1] = y_cote + frac * (Y_CONVOYEUR - y_cote)
            pos[i, 2] = Z_PRISE + 30 * (1 - frac * 0.5)
        else:
            frac = (ti - t1 - t2) / t3
            pos[i, 0] = X_DETECTION + V_CONVOYEUR * ti
            pos[i, 1] = Y_CONVOYEUR
            pos[i, 2] = Z_PRISE + 15 * (1 - frac)
    return t, pos

STRATEGIES = {'dessus': approche_dessus, 'laterale': approche_laterale}


# --- Verification de faisabilite ---

def verifier_faisabilite(p_init, p_cible, t_dispo):
    distance = np.linalg.norm(np.array(p_cible) - np.array(p_init))
    v_moyenne = distance / max(t_dispo, 1e-6)
    return v_moyenne < VMAX_ROBOT * 0.8, v_moyenne

def verifier_espace_travail(point):
    """Enveloppe simplifiee du TX2-90."""
    r = np.sqrt(point[0]**2 + point[1]**2)
    return 150 < r < 1000 and -200 < point[2] < 800


# --- Trajectoire de depot ---

def trajectoire_depot(p_prise, p_depot, duree=0.5, dt=0.001):
    """Trajectoire de la position de prise vers le depot (montee-translation-descente)."""
    t = np.arange(0, duree, dt)
    n = len(t)
    pos = np.zeros((n, 3))
    t1, t2, t3 = duree * 0.25, duree * 0.50, duree * 0.25
    z_secu = max(p_prise[2], p_depot[2]) + Z_APPROCHE

    for i in range(n):
        ti = t[i]
        if ti < t1:
            frac = ti / t1
            pos[i] = p_prise + frac * np.array([0, 0, z_secu - p_prise[2]])
        elif ti < t1 + t2:
            frac = (ti - t1) / t2
            p_h1 = np.array([p_prise[0], p_prise[1], z_secu])
            p_h2 = np.array([p_depot[0], p_depot[1], z_secu])
            pos[i] = p_h1 + frac * (p_h2 - p_h1)
        else:
            frac = (ti - t1 - t2) / t3
            pos[i] = np.array([p_depot[0], p_depot[1], z_secu + frac * (Z_DEPOT - z_secu)])
    return t, pos


# --- Cycle complet ---

def simuler_cycle_complet(objets_file, v_convoyeur=V_CONVOYEUR, strategie='dessus', dt=0.001):
    """Simule le cycle pick-and-place complet."""
    t_courant = 0.0
    p_robot = P_REPOS.copy()
    journal = []

    while True:
        obj = objets_file.prochain()
        if obj is None:
            break
        if obj.sorti_zone(t_courant, v_convoyeur):
            objets_file.marquer_rate(obj)
            journal.append({'t': t_courant, 'event': 'rate', 'objet': obj.id, 'raison': 'sorti de zone'})
            continue

        x_obj = obj.position_x(t_courant, v_convoyeur)
        if x_obj < X_PRISE_MIN:
            t_courant += (X_PRISE_MIN - x_obj) / v_convoyeur
        x_obj = obj.position_x(t_courant, v_convoyeur)
        t_sortie = (X_PRISE_MAX - x_obj) / v_convoyeur
        t_dispo = t_sortie * 0.7

        x_prise = obj.position_x(t_courant + t_dispo, v_convoyeur)
        p_prise = np.array([x_prise, Y_CONVOYEUR + obj.y_offset, Z_PRISE])

        faisable, v_moy = verifier_faisabilite(p_robot, p_prise, t_dispo)
        if not faisable:
            objets_file.marquer_rate(obj)
            journal.append({'t': t_courant, 'event': 'rate', 'objet': obj.id,
                             'raison': f'vitesse {v_moy:.0f} mm/s'})
            continue
        if not verifier_espace_travail(p_prise):
            objets_file.marquer_rate(obj)
            journal.append({'t': t_courant, 'event': 'rate', 'objet': obj.id,
                             'raison': 'hors espace'})
            continue

        fn_approche = STRATEGIES[strategie]
        t_traj, pos_traj = fn_approche(p_robot, p_prise, t_dispo, dt)

        t_prise = t_courant + t_dispo
        x_obj_reel = obj.position_x(t_prise, v_convoyeur)
        erreur_sync = abs(pos_traj[-1, 0] - x_obj_reel)
        p_reussite = probabilite_prise(erreur_sync)
        reussi = np.random.random() < p_reussite

        t_courant = t_prise + TEMPS_FERMETURE
        if reussi:
            objets_file.marquer_pris(obj)
            journal.append({'t': t_prise, 'event': 'pris', 'objet': obj.id,
                             'erreur_sync': erreur_sync, 'p_reussite': p_reussite})
            t_courant += 0.5 + TEMPS_OUVERTURE
            journal.append({'t': t_courant, 'event': 'depose', 'objet': obj.id})
            t_courant += np.linalg.norm(P_DEPOT - P_REPOS) / (VMAX_ROBOT * 0.6)
            p_robot = P_REPOS.copy()
        else:
            objets_file.marquer_rate(obj)
            journal.append({'t': t_prise, 'event': 'rate', 'objet': obj.id,
                             'raison': f'prise echouee (p={p_reussite:.2f})',
                             'erreur_sync': erreur_sync})
            p_robot = p_prise.copy()
            p_robot[2] = Z_APPROCHE

    return journal


# --- Metriques ---

def calculer_metriques_cycle(journal, n_objets):
    n_pris = sum(1 for e in journal if e['event'] == 'pris')
    taux = n_pris / max(n_objets, 1) * 100
    t_prises = [e['t'] for e in journal if e['event'] == 'pris']
    cycles = [t_prises[i] - t_prises[i-1] for i in range(1, len(t_prises))]
    t_cycle_moy = np.mean(cycles) if cycles else 0
    debit = 3600 / t_cycle_moy if t_cycle_moy > 0 else 0
    erreurs = [e.get('erreur_sync', 0) for e in journal if 'erreur_sync' in e]
    return {
        'n_pris': n_pris, 'taux_reussite': taux, 't_cycle_moy': t_cycle_moy,
        'debit_pieces_h': debit, 'erreur_sync_moy': np.mean(erreurs) if erreurs else 0,
    }


# --- Visualisation ---

def afficher_cycle(journal, file_objets, v_conv=V_CONVOYEUR):
    fig, axes = plt.subplots(2, 2, figsize=(14, 9))
    from matplotlib.patches import Patch

    # diagramme des evenements
    ax = axes[0, 0]
    couleurs_evt = {'pris': 'green', 'rate': 'red', 'depose': 'blue'}
    for evt in journal:
        if evt['event'] in couleurs_evt:
            ax.barh(evt['objet'], 0.1, left=evt['t'], color=couleurs_evt[evt['event']], height=0.6)
    ax.set_xlabel('Temps (s)'); ax.set_ylabel('Objet #')
    ax.set_title('Chronologie des evenements')
    ax.legend(handles=[Patch(color=c, label=n) for n, c in couleurs_evt.items()], fontsize=8)
    ax.grid(True, axis='x', alpha=0.3)

    # trajectoires des objets
    ax = axes[0, 1]
    t_max = max((e['t'] for e in journal), default=5.0)
    t_plot = np.linspace(0, t_max, 500)
    for obj in file_objets.historique:
        x_obj = [obj.position_x(ti, v_conv) for ti in t_plot]
        ax.plot(t_plot, x_obj, 'g-' if obj.pris else 'r--', linewidth=0.8, alpha=0.6)
    ax.axhspan(X_PRISE_MIN, X_PRISE_MAX, alpha=0.08, color='green')
    ax.set_xlabel('Temps (s)'); ax.set_ylabel('Position X (mm)')
    ax.set_title('Objets sur convoyeur'); ax.grid(True, alpha=0.3)

    # erreur de synchronisation par objet
    ax = axes[1, 0]
    erreurs = [(e['objet'], e.get('erreur_sync', 0), e['event'])
               for e in journal if 'erreur_sync' in e]
    if erreurs:
        ids, errs, evts = zip(*erreurs)
        couleurs = ['green' if ev == 'pris' else 'red' for ev in evts]
        ax.bar(ids, errs, color=couleurs, alpha=0.7)
        ax.axhline(y=TOLERANCE_PRISE, color='orange', linestyle='--',
                    label=f'Tolerance ({TOLERANCE_PRISE} mm)')
        ax.legend(fontsize=8)
    ax.set_xlabel('Objet #'); ax.set_ylabel('Erreur sync (mm)')
    ax.set_title('Erreur de synchronisation'); ax.grid(True, axis='y', alpha=0.3)

    # modele de probabilite
    ax = axes[1, 1]
    err_range = np.linspace(0, 2 * TOLERANCE_PRISE, 100)
    ax.plot(err_range, [probabilite_prise(e) for e in err_range], 'b-', linewidth=2)
    ax.axvline(x=TOLERANCE_PRISE, color='orange', linestyle='--')
    for e in journal:
        if 'erreur_sync' in e:
            c = 'green' if e['event'] == 'pris' else 'red'
            ax.plot(e['erreur_sync'], probabilite_prise(e['erreur_sync']), 'o', color=c, markersize=5)
    ax.set_xlabel('Erreur sync (mm)'); ax.set_ylabel('Probabilite')
    ax.set_title('Modele de probabilite de prise'); ax.grid(True, alpha=0.3)

    plt.suptitle('Prise a la volee -- Staubli TX2-90', fontsize=13, fontweight='bold')
    plt.tight_layout()
    return fig


# --- Programme principal ---

if __name__ == '__main__':
    np.random.seed(42)

    # generer les objets (intervalles irreguliers)
    file_obj = FileObjets()
    t_det = 0.0
    n_objets = 12
    for i in range(n_objets):
        t_det += 0.8 + np.random.exponential(0.5)
        y_off = np.random.uniform(-LARGEUR_CONVOYEUR * 0.3, LARGEUR_CONVOYEUR * 0.3)
        file_obj.ajouter(t_det, y_offset=y_off)

    print(f"Objets generes  : {n_objets}")
    print(f"V convoyeur     : {V_CONVOYEUR} mm/s")
    print(f"Zone de prise   : [{X_PRISE_MIN}, {X_PRISE_MAX}] mm\n")

    for strat in ['dessus', 'laterale']:
        np.random.seed(42)
        for obj in file_obj.historique:
            obj.pris, obj.rate = False, False
        journal = simuler_cycle_complet(file_obj, strategie=strat)
        m = calculer_metriques_cycle(journal, n_objets)
        print(f"--- Strategie : {strat} ---")
        print(f"  Pris          : {m['n_pris']}/{n_objets}")
        print(f"  Taux reussite : {m['taux_reussite']:.1f}%")
        print(f"  Temps cycle   : {m['t_cycle_moy']:.3f} s")
        print(f"  Debit         : {m['debit_pieces_h']:.0f} pieces/h")
        print(f"  Err sync moy  : {m['erreur_sync_moy']:.2f} mm\n")
        fig = afficher_cycle(journal, file_obj)
        fig.savefig(f'pick_fly_{strat}.png', dpi=150)
        plt.close(fig)

    # etude parametrique : influence de la vitesse
    print("=== Influence de la vitesse du convoyeur ===\n")
    vitesses = [100, 150, 200, 250, 300, 400]
    debits, taux = [], []
    for v in vitesses:
        np.random.seed(42)
        for obj in file_obj.historique:
            obj.pris, obj.rate = False, False
        journal = simuler_cycle_complet(file_obj, v_convoyeur=v, strategie='dessus')
        m = calculer_metriques_cycle(journal, n_objets)
        debits.append(m['debit_pieces_h'])
        taux.append(m['taux_reussite'])
        print(f"  V={v:4d} mm/s : taux={m['taux_reussite']:5.1f}%, debit={m['debit_pieces_h']:6.0f} p/h")

    fig2, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
    ax1.plot(vitesses, taux, 'bo-', linewidth=2)
    ax1.set_xlabel('V convoyeur (mm/s)'); ax1.set_ylabel('Taux (%)')
    ax1.set_title('Taux de reussite'); ax1.grid(True, alpha=0.3)
    ax2.plot(vitesses, debits, 'rs-', linewidth=2)
    ax2.set_xlabel('V convoyeur (mm/s)'); ax2.set_ylabel('Debit (p/h)')
    ax2.set_title('Debit'); ax2.grid(True, alpha=0.3)
    plt.suptitle('Etude parametrique -- Prise a la volee', fontweight='bold')
    plt.tight_layout()
    fig2.savefig('pick_fly_parametrique.png', dpi=150)
    plt.close(fig2)
    print("\nFigures sauvegardees : pick_fly_*.png")
