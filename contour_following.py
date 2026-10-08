"""
Suivi de contour avec regulation de force normale.
Simulation d'un outil suivant un profil piece avec asservissement
hybride position/force, controle en impedance, et adaptation des gains.
TP robotique industrielle -- Staubli TX2-90.
"""
import numpy as np
import matplotlib.pyplot as plt
import csv
from dataclasses import dataclass


# --- Profils de contour ---

def profil_sinusoidal(n_points=300):
    """Profil ondule classique, combine plusieurs harmoniques."""
    t = np.linspace(0, 1, n_points)
    x = t * 300
    y = (20 * np.sin(2 * np.pi * t)
         + 10 * np.sin(6 * np.pi * t)
         + 5 * np.cos(4 * np.pi * t + 0.5))
    return x, y

def profil_arc_circulaire(n_points=300, rayon=120, angle_deg=150):
    """Arc de cercle, typique d'un polissage sur piece cylindrique."""
    angle = np.deg2rad(angle_deg)
    theta = np.linspace(-angle / 2, angle / 2, n_points)
    x = rayon * np.sin(theta)
    y = rayon * (1 - np.cos(theta))
    x = x - x[0]
    return x, y

def profil_en_marches(n_points=300, n_marches=5, hauteur_marche=8.0):
    """Profil en escalier avec raccords, simule un usinage par paliers."""
    x = np.linspace(0, 300, n_points)
    y = np.zeros(n_points)
    largeur = 300.0 / n_marches
    for i in range(n_points):
        marche = min(int(x[i] / largeur), n_marches - 1)
        x_local = x[i] - marche * largeur
        if x_local < largeur * 0.05 and marche > 0:
            y[i] = (marche - 1 + x_local / (largeur * 0.05)) * hauteur_marche
        else:
            y[i] = marche * hauteur_marche
    return x, y

def profil_spline_libre(n_points=300):
    """Profil libre par interpolation de points de controle."""
    x_ctrl = np.array([0, 40, 80, 130, 170, 220, 260, 300])
    y_ctrl = np.array([0, 12, -5, 18, 8, -10, 15, 3])
    x = np.linspace(0, 300, n_points)
    y = np.interp(x, x_ctrl, y_ctrl)
    noyau = 15
    y_lisse = np.convolve(y, np.ones(noyau) / noyau, mode='same')
    return x, y_lisse

PROFILS = {
    'sinusoidal': profil_sinusoidal,
    'arc': profil_arc_circulaire,
    'marches': profil_en_marches,
    'spline': profil_spline_libre,
}


# --- Geometrie du profil ---

def normale_au_profil(x, y):
    """Calcule les vecteurs normaux au profil (vers l'exterieur)."""
    dx = np.gradient(x)
    dy = np.gradient(y)
    longueur = np.sqrt(dx**2 + dy**2)
    return -dy / longueur, dx / longueur

def courbure_profil(x, y):
    """Courbure locale du profil (inverse du rayon de courbure)."""
    dx, dy = np.gradient(x), np.gradient(y)
    ddx, ddy = np.gradient(dx), np.gradient(dy)
    num = np.abs(dx * ddy - dy * ddx)
    return num / np.maximum((dx**2 + dy**2)**1.5, 1e-10)


# --- Modele de capteur d'effort ---

@dataclass
class CapteurEffort:
    """Modele realiste : biais, derive, bruit, quantification, poids outil."""
    biais: float = 0.15
    sigma_bruit: float = 0.05
    derive_par_s: float = 0.01
    resolution: float = 0.01   # pas de quantification (N)
    masse_outil: float = 0.8
    g: float = 9.81

    def mesurer(self, force_reelle, t, angle_normal=0.0):
        """Mesure bruitee avec compensation du poids outil."""
        poids_projete = self.masse_outil * self.g * np.cos(angle_normal)
        brut = force_reelle + poids_projete + self.biais
        brut += self.derive_par_s * t
        brut += np.random.normal(0, self.sigma_bruit)
        brut = np.round(brut / self.resolution) * self.resolution
        return brut, poids_projete


# --- Detection de contact ---

def detecter_contact(forces, seuil=1.5, n_confirmation=5):
    """Detecte le premier contact confirme sur n_confirmation echantillons."""
    compteur = 0
    for i, f in enumerate(forces):
        if f > seuil:
            compteur += 1
            if compteur >= n_confirmation:
                return i - n_confirmation + 1
        else:
            compteur = 0
    return -1

def detecter_perte_contact(forces, seuil=0.5, n_confirmation=3):
    """Detecte une perte de contact (force sous le seuil)."""
    compteur = 0
    for i, f in enumerate(forces):
        if f < seuil:
            compteur += 1
            if compteur >= n_confirmation:
                return i - n_confirmation + 1
        else:
            compteur = 0
    return -1


# --- Controleurs ---

def gains_adaptatifs(courbure, kp_base, ki_base, kd_base):
    """Adaptation des gains PID selon la courbure locale."""
    facteur = 1.0 / (1.0 + 50.0 * courbure)
    return kp_base * facteur, ki_base * facteur, kd_base * (2.0 - facteur)

def controle_impedance(force_mesuree, f_consigne, vitesse,
                       amortissement=10.0, raideur_v=50.0):
    """Loi de commande en impedance : M*a + B*v + K*x = f_ext - f_des."""
    erreur_f = force_mesuree - f_consigne
    return (erreur_f - amortissement * vitesse) / raideur_v


# --- Simulation principale ---

def simuler_suivi_contour(x_profil, y_profil, f_consigne=10.0,
                           raideur_contact=50.0, kp=0.5, ki=0.02, kd=0.1,
                           dt=0.005, mode='pid_adaptatif', vitesse_avance=60.0):
    """
    Simulation du suivi de contour avec regulation de force.
    mode : 'pid' / 'pid_adaptatif' / 'impedance'
    """
    n = len(x_profil)
    nx, ny = normale_au_profil(x_profil, y_profil)
    courbures = courbure_profil(x_profil, y_profil)
    capteur = CapteurEffort()
    offset_nominal = f_consigne / raideur_contact
    x_outil, y_outil = np.zeros(n), np.zeros(n)
    erreur_int, erreur_prec, vitesse_normale = 0.0, 0.0, 0.0
    forces, corrections = np.zeros(n), np.zeros(n)
    en_contact = np.zeros(n, dtype=bool)

    for i in range(n):
        t_courant = i * dt
        if i == 0:
            x_outil[i] = x_profil[i] + nx[i] * offset_nominal
            y_outil[i] = y_profil[i] + ny[i] * offset_nominal
            forces[i] = f_consigne
            en_contact[i] = True
            continue

        raideur_locale = raideur_contact * (1 + 0.3 * np.sin(2 * np.pi * i / n * 5))
        dist = np.sqrt((x_outil[i-1] - x_profil[i])**2
                        + (y_outil[i-1] - y_profil[i])**2)
        force_reelle = raideur_locale * max(0, offset_nominal - dist)
        angle_n = np.arctan2(nx[i], ny[i])
        force_brute, poids = capteur.mesurer(force_reelle, t_courant, angle_n)
        force_mesuree = max(force_brute - poids - capteur.biais, 0)
        forces[i] = force_mesuree
        en_contact[i] = force_mesuree > 0.5

        # feedforward base sur la courbure
        dx_p, dy_p = np.gradient(x_profil)[i], np.gradient(y_profil)[i]
        feedforward = 0.001 * np.sqrt(dx_p**2 + dy_p**2) * vitesse_avance * courbures[i]

        if mode == 'pid':
            erreur = f_consigne - force_mesuree
            erreur_int = np.clip(erreur_int + erreur * dt, -5, 5)
            correction = kp * erreur + ki * erreur_int + kd * (erreur - erreur_prec) / dt + feedforward
            erreur_prec = erreur
        elif mode == 'pid_adaptatif':
            kp_a, ki_a, kd_a = gains_adaptatifs(courbures[i], kp, ki, kd)
            erreur = f_consigne - force_mesuree
            erreur_int = np.clip(erreur_int + erreur * dt, -5, 5)
            correction = kp_a * erreur + ki_a * erreur_int + kd_a * (erreur - erreur_prec) / dt + feedforward
            erreur_prec = erreur
        elif mode == 'impedance':
            correction = controle_impedance(force_mesuree, f_consigne, vitesse_normale) + feedforward
        else:
            raise ValueError(f"Mode inconnu : {mode}")

        vitesse_normale = correction / dt if dt > 0 else 0
        offset_corrige = np.clip(offset_nominal + correction, 0.01, 3 * offset_nominal)
        x_outil[i] = x_profil[i] + nx[i] * offset_corrige
        y_outil[i] = y_profil[i] + ny[i] * offset_corrige
        corrections[i] = correction

        # recuperation si perte de contact
        if not en_contact[i] and i > 10:
            x_outil[i] = x_profil[i] + nx[i] * offset_nominal * 0.5
            y_outil[i] = y_profil[i] + ny[i] * offset_nominal * 0.5
            erreur_int = 0.0

    return x_outil, y_outil, forces, corrections, en_contact


# --- Metriques de performance ---

def calculer_metriques(forces, f_consigne, dt=0.005):
    """Metriques de performance du suivi de force."""
    erreur = forces - f_consigne
    mae = np.mean(np.abs(erreur))
    rms = np.sqrt(np.mean(erreur**2))
    max_err = np.max(np.abs(erreur))
    bande = f_consigne * 0.05
    t_etablissement = None
    for i in range(len(erreur) - 1, -1, -1):
        if abs(erreur[i]) > bande:
            t_etablissement = (i + 1) * dt
            break
    return {'mae': mae, 'max_err': max_err, 'rms': rms, 't_etablissement': t_etablissement}

def exporter_csv(chemin, x_profil, y_profil, x_outil, y_outil, forces, f_consigne):
    """Exporte les resultats dans un fichier CSV."""
    with open(chemin, 'w', newline='') as f:
        writer = csv.writer(f, delimiter=';')
        writer.writerow(['x_profil_mm', 'y_profil_mm', 'x_outil_mm', 'y_outil_mm',
                          'force_N', 'erreur_force_N'])
        for i in range(len(x_profil)):
            writer.writerow([f'{x_profil[i]:.3f}', f'{y_profil[i]:.3f}',
                              f'{x_outil[i]:.3f}', f'{y_outil[i]:.3f}',
                              f'{forces[i]:.3f}', f'{forces[i] - f_consigne:.3f}'])


def campagne_simulation(nom_profil='sinusoidal', n_runs=5, f_consigne=10.0):
    """Lance plusieurs runs avec parametres legerement differents."""
    generateur = PROFILS[nom_profil]
    resultats_runs = []
    for run in range(n_runs):
        np.random.seed(run * 42 + 7)
        x_p, y_p = generateur(300)
        raideur = 50.0 * (1 + 0.2 * (np.random.random() - 0.5))
        x_o, y_o, forces, corr, contact = simuler_suivi_contour(
            x_p, y_p, f_consigne=f_consigne, raideur_contact=raideur,
            mode='pid_adaptatif')
        m = calculer_metriques(forces, f_consigne)
        m['raideur'] = raideur
        resultats_runs.append(m)
    return resultats_runs


# --- Visualisation ---

def afficher_resultats(x_p, y_p, x_o, y_o, forces, corrections,
                        en_contact, f_cible, metriques, titre_profil=''):
    """Affichage 4 sous-graphiques avec annotations."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 9))
    erreur_force = forces - f_cible

    ax = axes[0, 0]
    ax.plot(x_p, y_p, 'k-', linewidth=2, label='Profil piece')
    ax.plot(x_o, y_o, 'r-', linewidth=1, alpha=0.7, label='Trajectoire outil')
    if np.any(~en_contact):
        ax.plot(x_o[~en_contact], y_o[~en_contact], 'yo', markersize=3, label='Perte contact')
    ax.set_xlabel('X (mm)'); ax.set_ylabel('Y (mm)')
    ax.set_title(f'Suivi de contour ({titre_profil})')
    ax.legend(fontsize=8); ax.grid(True, alpha=0.3); ax.set_aspect('equal')

    ax = axes[0, 1]
    ax.plot(x_p, forces, 'b-', linewidth=0.8, alpha=0.8)
    ax.axhline(y=f_cible, color='r', linestyle='--', label=f'Consigne ({f_cible} N)')
    ax.fill_between(x_p, f_cible * 0.95, f_cible * 1.05, alpha=0.1, color='green', label='Bande 5%')
    ax.set_xlabel('Position X (mm)'); ax.set_ylabel('Force (N)')
    ax.set_title('Force normale'); ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

    ax = axes[1, 0]
    ax.plot(x_p, erreur_force, 'g-', linewidth=0.8)
    ax.axhline(y=0, color='k', linewidth=0.5)
    txt = f'MAE={metriques["mae"]:.2f} N\nRMS={metriques["rms"]:.2f} N\nMax={metriques["max_err"]:.2f} N'
    ax.text(0.02, 0.95, txt, transform=ax.transAxes, fontsize=8,
            verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    ax.set_xlabel('Position X (mm)'); ax.set_ylabel('Erreur (N)')
    ax.set_title('Erreur de force'); ax.grid(True, alpha=0.3)

    ax = axes[1, 1]
    ax.plot(x_p, corrections, 'm-', linewidth=0.8, label='Correction PID')
    courbures = courbure_profil(x_p, y_p)
    ax2 = ax.twinx()
    ax2.plot(x_p, courbures * 1000, 'c-', linewidth=0.5, alpha=0.5, label='Courbure (x1000)')
    ax2.set_ylabel('Courbure (1/m)', color='c')
    ax.set_xlabel('Position X (mm)'); ax.set_ylabel('Correction (mm)')
    ax.set_title('Sortie correcteur et courbure')
    ax.legend(loc='upper left', fontsize=8); ax2.legend(loc='upper right', fontsize=8)
    ax.grid(True, alpha=0.3)

    plt.suptitle('Suivi de contour avec regulation de force -- Staubli TX2-90',
                  fontsize=13, fontweight='bold')
    plt.tight_layout()
    return fig


# --- Programme principal ---

if __name__ == '__main__':
    f_cible = 10.0
    n_pts = 300

    # comparer les trois modes sur le profil sinusoidal
    print("=== Comparaison des modes de controle ===\n")
    x_p, y_p = profil_sinusoidal(n_pts)
    for mode in ['pid', 'pid_adaptatif', 'impedance']:
        np.random.seed(123)
        x_o, y_o, forces, corr, contact = simuler_suivi_contour(
            x_p, y_p, f_consigne=f_cible, mode=mode)
        m = calculer_metriques(forces, f_cible)
        n_perte = np.sum(~contact[10:])
        print(f"  Mode {mode:18s} : MAE={m['mae']:.3f} N, "
              f"RMS={m['rms']:.3f} N, Max={m['max_err']:.3f} N, "
              f"Pertes={n_perte}")

    # simulation par profil avec le mode adaptatif
    print("\n=== Simulation par profil (mode adaptatif) ===\n")
    for nom, generateur in PROFILS.items():
        np.random.seed(42)
        x_p, y_p = generateur(n_pts)
        x_o, y_o, forces, corr, contact = simuler_suivi_contour(
            x_p, y_p, f_consigne=f_cible, mode='pid_adaptatif')
        m = calculer_metriques(forces, f_cible)
        print(f"  Profil {nom:12s} : MAE={m['mae']:.3f} N, RMS={m['rms']:.3f} N")
        fig = afficher_resultats(x_p, y_p, x_o, y_o, forces, corr,
                                  contact, f_cible, m, titre_profil=nom)
        fig.savefig(f'contour_{nom}.png', dpi=150)
        plt.close(fig)

    exporter_csv('contour_resultats.csv', x_p, y_p, x_o, y_o, forces, f_cible)
    print("\nResultats exportes : contour_resultats.csv")

    # campagne statistique
    print("\n=== Campagne statistique (10 runs, profil sinusoidal) ===\n")
    stats = campagne_simulation('sinusoidal', n_runs=10, f_consigne=f_cible)
    maes = [s['mae'] for s in stats]
    rms_vals = [s['rms'] for s in stats]
    print(f"  MAE : moyenne={np.mean(maes):.3f}, ecart-type={np.std(maes):.3f}")
    print(f"  RMS : moyenne={np.mean(rms_vals):.3f}, ecart-type={np.std(rms_vals):.3f}")
    print("\nFigures sauvegardees : contour_*.png")
