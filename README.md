# Suivi de contour et commande en effort -- Staubli TX2-90

Travaux pratiques de robotique industrielle sur le robot Staubli TX2-90, realises a l'ENSMM Besancon. Programmation en Val 3 et simulation sous Staubli Robotics Suite (SRS). Les travaux couvrent le suivi de contour avec controle de force, la prise a la volee sur convoyeur, et la synchronisation d'axes.

## Contenu

- `contour_following.py` : simulation du suivi de contour avec regulation de force normale, discretisation de la trajectoire le long d'un profil quelconque
- `pick_on_the_fly.py` : calcul de la trajectoire de prise a la volee sur un convoyeur en mouvement, synchronisation vitesse robot / convoyeur
- `axis_sync.py` : synchronisation d'axes pour un deplacement coordonne multi-robot ou robot + axe externe
- `force_control.py` : boucle de regulation PID en effort, simulation de l'interaction outil-piece avec raideur de contact

## Notions couvertes

- Commande en effort (force control) avec capteur force/couple
- Suivi de trajectoire avec asservissement hybride position/force
- Prise a la volee (pick on the fly) sur convoyeur
- Synchronisation d'axes (tracking convoyeur)
- Profils de vitesse trapezoIdaux
- Interpolation circulaire et lineaire (MoveL, MoveC)

## Dependances

- Python 3.8+
- numpy, matplotlib, scipy
