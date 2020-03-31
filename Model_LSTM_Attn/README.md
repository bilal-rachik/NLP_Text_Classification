## Modèle Seq2Seq avec mécanisme d'attention

Implémentation du modèle Seq2Seq avec mécanisme d'attention
[Neural Machine Translation by Jointly Learning to Align and Translate](https://arxiv.org/pdf/1409.0473.pdf), [Text Classification Research with Attention-based Recurrent Neural Networks](http://univagora.ro/jour/index.php/ijccc/article/download/3142/pdf)


Les modèles Seq2Seq ont été largement utilisés dans des problèmes tels que la traduction 
automatique, le résumé de documents en raison de leur capacité à générer une nouvelle séquence 
basée sur les données déjà vues. Ici, nous avons implémenté le modèle Seq2Seq pour la tâche de 
classification de texte.

L'architecture du modèle Seq2Seq Attention pour la classification.

Architecture d'attention Seq2Seq

![TextRNN Architecture](data/BiLSTM-Attention.ppm)

## Détails d'implémentation

* fastText fr Embeddings pré-formés utilisés pour initialiser les vecteurs de mots
* 2 couches de BiLSTM
* Utilisé 64 unités cachées dans chaque couche BiLSTM
* Dropou avec probabilité de conservation 0,4
* Optimiseur - adam
* Fonction de perte - entropie croisée (CrossEntropyLoss)
* Séquences de longueur 30