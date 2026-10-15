# Suivi de contour et regulation de force -- Staubli TX2-90

Programmes VAL3 developpes dans le cadre des TP de robotique industrielle sur cellule Staubli TX2-90 a l'ENSMM.

## Contenu

### contourSuivi.pgx
Suivi de contour avec regulation de force normale. Asservissement hybride position/force avec adaptation des gains PID en fonction de la courbure locale du profil piece. Plusieurs types de profils (sinusoidal, arc, marches, spline). Compensation du poids outil et detection de perte de contact.

### regulationForce.pgx
Regulation PID en effort avec comparaison de plusieurs structures de correcteurs (P, PI, PID, PID + feedforward). Auto-reglage par methode de Ziegler-Nichols en boucle fermee. Predicteur de Smith pour la compensation du retard pur. Gestion de l'anti-windup et derivee filtree.

### pickOnTheFly.pgx
Prise a la volee sur convoyeur avec suivi par encodeur incremental. Deux strategies d'approche (par le dessus, laterale). Cycle complet pick-and-place avec verification de faisabilite cinematique, modele probabiliste de reussite, et synchronisation robot-convoyeur.

### syncAxes.pgx
Synchronisation multi-axes pour coordination robot + axe externe (rail lineaire). Profils de vitesse trapezoidal et en S (limitation du jerk). Raccordement en zone fine entre segments. Engrenage electronique, profil de came, et compensation du jeu mecanique (backlash) dans les reducteurs.

## Environnement

- Robot : Staubli TX2-90
- Controleur : CS9 / VAL3
- Capteur d'effort : 6 axes
- Pince : Robotiq 2F-85
- Convoyeur avec encodeur incremental
- Axe externe lineaire
- Plateforme FiRAC, ENSMM / Etupes

## Competences

- Programmation VAL3 (langage natif Staubli)
- Regulation PID industrielle et controle en impedance
- Synchronisation multi-axes et profils de mouvement
- Suivi de contour avec capteur d'effort
- Pick on the fly avec suivi encodeur convoyeur
