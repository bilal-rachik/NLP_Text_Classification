# NLP_Text_Classification
Ce référentiel contient quelques bonnes pratiques d'apprentissage automatique clés pour résoudre
les problèmes de classification de texte. 

Voici ce que vous apprendrez:

* Le flux de travail de haut niveau de bout en bout pour résoudre les problèmes de classification de texte à l'aide de l'apprentissage automatique
* Comment choisir le bon modèle pour votre problème de classification de texte
* Comment implémenter votre modèle de choix
*  L'implémentation de divers modèles de classification de texte



## Performance du modèle

<table>
  <tr>
    <th rowspan="3">Model</th>
    <th align="center" colspan="14">nb_line </th>
  </tr> 
  <tr>
    <th colspan="7">50000</th>
    <th colspan="7">100000</th>
  </tr>
  <tr>
    <th>Accuracy</th>
    <th>precision weighted avg</th>
    <th>recall weighted avg</th>
    <th>f1-score weighted avg </th>
     <th>precision macro avg </th>
    <th>recall macro avg </th>
    <th>f1-score macro avg   </th>
    <th>Accuracy</th>
    <th>precision weighted  avg </th>
    <th>recall weighted  avg </th>
    <th>f1-score weighted  avg   </th>
    <th>precision macro avg </th>
    <th>recall macro avg </th>
    <th>f1-score macro avg   </th>
      
      
  </tr>
  <tr>
    <td>Bagging</td>
    <td> 0.78</td>
    <td> 0.78</td>
    <td> 0.78</td>
    <td> 0.77</td>
    <td>  0.53</td>
    <td> 0.41</td>
    <td> 0.44</td>
    <td> 0.81</td>
    <td> 0.81</td>
    <td> 0.81</td>
    <td> 0.80</td>
    <td>  0.64</td>
    <td> 0.42</td>
    <td> 0.52</td>
  </tr>
  <tr>
    <td>Random Forest</td>
    <td>0.81 </td>
    <td> 0.81</td>
    <td> 0.81</td>
    <td> 0.79</td>
    <td> 0.63</td>
    <td> 0.49</td>
    <td> 0.53</td>
    <td> 0.83</td>
    <td> 0.83</td>
    <td> 0.83</td>
    <td>0.83 </td>
    <td> 0.73 </td>
    <td> 0.55</td>
    <td> 0.60</td>
  </tr>
  <tr>
    <td>AdaBoost</td>
    <td> 0.77 </td>
    <td> 0.76</td>
    <td> 0.77</td>
    <td>  0.76</td>
    <td> 0.53</td>
    <td> 0.40</td>
    <td> 0.44</td>
    <td> 0.81</td>
    <td> 0.81</td>
    <td> 0.81</td>
    <td> 0.80</td>
    <td>  0.63</td>
    <td> 0.48</td>
    <td> 0.53</td>
  </tr>
  <tr>
    <td>gradient boosting</td>
    <td> 0.79</td>
    <td> 0.79</td>
    <td> 0.79</td>
    <td> 0.78</td>
    <td> 0.61</td>
    <td> 0.54</td>
    <td> 0.65</td>
    <td> 0.80</td>
    <td> 0.81</td>
    <td> 0.80</td>
    <td> 0.79</td>
    <td> 0.65</td>
    <td> 0.57</td>
    <td>0.60 </td>
  </tr>
  <tr>
    <td>régression logistique</td>
    <td> 0.87</td>
    <td> 0.88</td>
    <td> 0.87</td>
    <td> 0.88</td>
    <td> 0.68</td>
    <td> 0.65</td>
    <td> 0.65</td>
     <td> 0.88</td>
    <td> 0.89</td>
    <td> 0.88</td>
    <td>0.88 </td>
    <td> 0.73</td>
    <td>0.70</td>
    <td>0.69 </td>
  </tr>
  <tr>
    <td>Naive Bayes</td>
    <td>0.72 </td>
    <td> 0.75</td>
    <td> 0.72</td>
    <td> 0.72</td>
    <td> 0.52</td>
    <td> 0.25</td>
    <td> 0.28</td>
     <td> 0.71</td>
    <td> 0.79</td>
    <td> 0.75</td>
    <td> 0.71</td>
     <td> 0.66</td>
    <td> 0.30</td>
    <td> 0.34</td>
  </tr>
  <tr>
    <td>SVM</td>
    <td>0.88 </td>
    <td> 0.89</td>
    <td> 0.88</td>
    <td> 0.89</td>
    <td> 0.70</td>
    <td> 0.69</td>
    <td> 0.68</td>
    <td> 0.89</td>
    <td> 0.90</td>
    <td> 0.89</td>
    <td> 0.90</td>
    <td> 0.69</td>
    <td> 0.70</td>
    <td> 0.69</td>
  </tr>
   <tr>
    <td>MLP</td>
    <td> 0.89</td>
    <td> 0.89</td>
    <td> 0.89</td>
    <td> 0.89</td>
    <td> 0.73</td>
    <td> 0.60</td>
    <td> 0.64</td>
    <td> 0.90</td>
    <td> 0.90</td>
    <td> 0.90</td>
    <td> 0.90</td>
    <td> 0.75</td>
    <td> 0.65</td>
    <td> 0.68</td>
  </tr>
</table>