# Määrittelydokumentti

Projekti toteutetaan Pythonilla.

Vertaisarvioin mieluiten Pythonilla toeutettuja projekteja, mutta myös Javalla toteutetun projektin arviointi saattaa onnistua.

Projektin idea on kouluttaa neroverkko tunnistamaan käsinkirjoitettuja numeroita MNIST käsinkirjoitettujen numeroiden datasetillä.

Tietorakenteina käytetään numpy-kirjaston taulukoita. Algoritmina gradient descent -algoritmi.

Ohjelma saa syötteenä kuvan, joka on muutettu vektoriksi, jonka alkiot ovat arvoja väliltä 0-1 pikseleiden kirkkauden perusteella.
Syötteen perusteella ohjelma yrittää luokitella kvuan yhteen kymmenestä luokasta (numero väliltä 0-9).
Koulutuksen aikana Ohjelmalle annetaan syötteenä myös luokka, johon ohjelman pitäisi saada kuva luokiteltua.

Algoritmista on tarkoitus saada niin tehokas, että se osaa luokitella kuvan välittömästi.

Aion käyttää lähteinä 3blue1brown:in [videosarjaa](https://www.3blue1brown.com/topics/neural-networks),
sekä [Mathematics for Machine Learning](https://mml-book.github.io)-kirjaa.

Harjoitustyön ydin on itse neuroverkko, joka hoitaa laskennan ja syötteen luokittelun.
Ytimeen kuuluu myös neuroverkon kouluttamiseen vaadittavat osat.

Opinto-ohjelmani on tietojenkäsittelyn kandidaatti.

Dokumentaatio on tarkoitus kirjoittaa muuten suomeksi,
mutta koodiin sijoitettavien kommenttien kirjoittaminen englanniksi ei liene ongelma?
