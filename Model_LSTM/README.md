# RNN (BidLSTM) Model
Ici, nous avons implémenté un réseau bidirectionnelle à long terme (BIDRNN) PyTorch.

Les LSTM ont été très populaires pour résoudre les problèmes de classification de texte en raison de leur propriété théorique de capturer tout le contexte tout en représentant une phrase.
## Architecture du modèle
L'architecture du LSTM bidirectionnel est la suivante:


![TextRNN Architecture](data/BILSTM.jpeg)

## Détails d'implémentation
* Embaddings FastText pré-formés utilisés pour initialiser les vecteurs de mots
* 2 couches de RNN
* Utilisé 64 unités cachées dans chaque couche BiLSTM
* dropout 0.4
* Optimiseur  adam 
* Fonction de perte - entropie croisée (CrossEntropyLoss)
*  longueurs de séquence 30
e 50ibles et des séquences de longueur 50