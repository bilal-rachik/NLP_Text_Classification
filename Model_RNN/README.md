# RRNN (BidRNN) Model
Ici, nous avons implémenté un réseau RNN bidirectionnelle avec  PyTorch.


## Architecture du modèle
L'architecture du RNN bidirectionnel est la suivante:


![TextRNN Architecture](data/RNN_V.jpg)


## Détails d'implémentation
* Embaddings FastText pré-formés utilisés pour initialiser les vecteurs de mots
* 2 couches de BiLSTM
* Utilisé 64 unités cachées dans chaque couche BiLSTM
* dropout 0.4
* Optimiseur  adam 
* Fonction de perte - entropie croisée (CrossEntropyLoss)
*  longueurs de séquence 50ib


