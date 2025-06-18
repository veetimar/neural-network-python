# Käyttöohje

Asenna tarvittavat riippuvuudet

```console
poetry install --without dev
```

Suorita ohjelma

```console
poetry run python src/digits.py
```

Suorittamisen jälkeen sinun tulee syöttää ohjelmalle teksti "train", "test" tai "exit". Exit sulkee ohjelman,
train siirtyy koulutukseen ja test testaa verkon kykyä tunnistaa käsinkirjoitettuja numeroita.

Valitessasi "train", ohjelma alkaa kysellä hyperparametreja koulutusta varten. Kysymyksien yhteydessä näkyy myös suluilla vakioarvo,
jota käytetään jättäessäsi kentän tyhjäksi. Koulutuksen alkaessa tulee vain odottaa, että koulutus saadaan loppuun.
Kun koulutus on päättynyt, ohjelma piirtää ruudulle käyrän siitä, kuinka hyvin ohjelma keskimäärin suoriutui jokaisella kerralla,
kun dataset käytiin kokonaisuudessaan läpi. Tarkasteltuasi tätä voit sulkea ikkunan, jolloin palataan takaisin päävalikkoon.

Valitessasi "test", ohjelma kokeilee, kuinka monta testikuvaa neuroverkko tunnistaa oikein ja ilmoittaa sen käyttäjälle.
Testauksen jälkeen käyttäjällä on mahdollisuus piirtää ruudulle numerot, jotka neuroverkko tunnisti väärin komennolla "w" tai
vaihtoehtoisesti numerot, jotka neuroverkko tunnisti oikein komennolla "r". Tämän voi myös ohittaa painamalla ainoastaan enteriä,
jolloin palataan jälleen päävalikkoon. Tarkastellessa kuvia voi seuraavaan kuvaan siirtyä syöttämällä komentoriviin komennon "f"
ja edelliseen komennolla "d". Myös tarkastelusta poistutaan enterillä.
