#Modèle CharCNN
Il s'agit de la mise en œuvre de CNN au niveau des caractères comme proposé dans l'article [Character-level Convolutional Networks for Text
Classification](https://papers.nips.cc/paper/5782-character-level-convolutional-networks-for-text-classification.pdf).  .

Dans CharCNN, le texte d'entrée est représenté par une matrice ( l_0 , d ). Où l_0 est la longueur de phrase maximale et d est la dimensionnalité de l'incorporation de caractères.

Les caractères suivants sont utilisés pour la quantification des caractères:

abcdefghijklmnopqrstuvwxyz0123456789 -,;.!?: '' '/ \ | _ @ # $% ˆ & * ̃' + - = <> () [] {}

Architecture du modèle

![TextRNN Architecture](data/convnet.png)