# Testausdokumentti

## Network-luokan testaus:

Kaikki metodit, jotka palauttavat jotakin on yksikkötestattu. Yksikkötesteissä varmistetaan, että metodin palautus on oikeaa muotoa
(numpy-taulukko on oikean muotoinen, lista on oikean pituinen tai palautus on varmasti yksittäinen luku).
Joissakin metodeissa oletetaan aina tietyn muotoista taulukkoa palautettavaksi, ja joissakin taas esimerkiksi
palautettavan taulukon oletetaan olevan saman muotuoinen, kuin parametrina annettu taulukko.
Sellaisissa metodeissa, joissa näin on mielekästä toimia, on myös testattu että palautettavat arvot ovat oikein.
Näissä tapauksissa on etukäteen laskettu, mitä metodin pitäisi palauttaa annetuilla syötteillä ja varmistetaan,
että metodi todella palauttaa kyseiset arvot.

Lisäksi on testattu, että luokan konstruktori asettaa attribuuttina olevien listojen pituuden oikein
(vastaa neuroverkon kerroksien määrää) ja listoissa olevien paino-matriisien koot ovat oikein
(vastaavat neuronien määrää kerroksilla).

Painojen tallentamista ja lataamista tiedostosta on testattu tallentamalla verkon painot tiedostoon,
luomalla uusi verkko, lataamalla uuteen verkkoon painot samasta tiedostosta
ja vertaamalla että verkkojen painot ovat tämän jälkeen täysin samalaiset.

Testeissä varmistetaan myös, että neuroverkko oppii täysin (overfit) xor-portin toiminnan,
kaikki painot muuttuvat koulutuksen aikana ja backpropagation ei vaikuta input-kerroksen attribuutteihin.

Neuroverkko saavuttaa MNIST-käsinkirjoitettujen numeroiden datasetissä noin 97% tarkkuuden käymällä koulutusdata 10 kertaa läpi
(1,5 minuutin koulutuksella). Lisäksi käymällä koulutusdata vain kerran läpi (noin 10 sekunnin koulutuksella)
saavutetaan jo noin 94% tarkkuus. Jo tästä voidaan päätellä, että neuroverkko toimii suurinpiirtein oikein.

Mainittu noin 97% tarkkuus saavutetaan kouluttamalla neuroverkkoa kerran ohjelman itse ehdottamilla hyperparametreilla.
Pienempi tarkkuus saavutetaan kouluttamalla verkkoa kerran seuraavilla hyperparametreilla:

- Epochs 1
- Batch size 1
- Learning rate 0.1

Myös yli 98% tarkkuus on saavutettu,
mutta vain lisäämällä verkolle yksi ylimääräinen 112:n neuronin kerros heti ensimmäisen kerroksen jälkeen,
jolloin koulutus muuttui kuitenkin jo häiritsevän hitaaksi vaaditun laskentatehon kasvaessa huomattavasti.

Testikattavuutta mitattu coverage-työkalulla. Kattavuus 100%
