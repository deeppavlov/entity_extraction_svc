## Entity Extraction Service
Entity extraction API powered by DeepPavlov configs.


## Run with docker-compose
`docker-compose up --build`


## How to
Main gateway (with Swagger UI) is available at http://localhost:9999/


#### POST `/`
```json
{
  "text": "The Mona Lisa is a sixteenth century oil painting created by Leonardo. It's held at the Louvre in Paris.."
}
```

#### RESPONSE
<details>

<summary>Show full response</summary>

```json
{
  "annotations":[
    {
      "start":0,
      "end":13,
      "spot":"The Mona Lisa",
      "confidence":1.0,
      "id":"Q2126369",
      "title":"Mona Lisa (Prado)",
      "uri":"https://en.wikipedia.org/wiki/Mona_Lisa_(Prado)",
      "abstract":"The Prado Mona Lisa is a painting by the workshop of Leonardo da Vinci and depicts the same subject and composition as Leonardos better known Mona Lisa at the Louvre, Paris. The Prado Mona Lisa has been in the collection of the Museo del Prado in Madrid, Spain since 1819, but was considered for decades a relatively unimportant copy. Following its restoration in 2012, however, the Prados Mona Lisa has come to be understood as the earliest known studio copy of Leonardos masterpiece.",
      "label":"Mona Lisa (Prado)",
      "categories":[
        "Mona Lisa",
        "Paintings of the Museo del Prado by Italian artists"
      ],
      "tags":[
        "WORK_OF_ART",
        "LITERARY_WORK",
        "FAC"
      ],
      "types":[
        
      ],
      "image":{
        "full":"https://commons.wikimedia.org/wiki/Special:FilePath/Gioconda_(copia_del_Museo_del_Prado_restaurada).jpg",
        "thumbnail":"https://commons.wikimedia.org/wiki/Special:FilePath/Gioconda_(copia_del_Museo_del_Prado_restaurada).jpg?width=300"
      },
      "lod":{
        "wikipedia":"https://en.wikipedia.org/wiki/Mona_Lisa_(Prado)"
      },
      "extras":[
        
      ]
    },
    {
      "start":61,
      "end":69,
      "spot":"Leonardo",
      "confidence":1.0,
      "id":"Q762",
      "title":"Leonardo da Vinci",
      "uri":"https://en.wikipedia.org/wiki/Leonardo_da_Vinci",
      "abstract":"Leonardo di ser Piero da Vinci (15 April 14522 May 1519) was an Italian polymath of the High Renaissance who was active as a painter, draughtsman, engineer, scientist, theorist, sculptor and architect. While his fame initially rested on his achievements as a painter, he also became known for his notebooks, in which he made drawings and notes on a variety of subjects, including anatomy, astronomy, botany, cartography, painting, and paleontology. Leonardos genius epitomized the Renaissance humanist ideal, and his collective works compose a contribution to later generations of artists matched only by that of his younger contemporary, Michelangelo.",
      "label":"Leonardo da Vinci",
      "categories":[
        "Leonardo da Vinci",
        "1452 births",
        "1519 deaths",
        "15th-century Italian mathematicians",
        "15th-century Italian painters",
        "15th-century Italian scientists",
        "15th-century Italian sculptors",
        "15th-century people of the Republic of Florence",
        "16th-century Italian mathematicians",
        "16th-century Italian painters"
      ],
      "tags":[
        "PER",
        "WRITER",
        "BUSINESS"
      ],
      "types":[
        "https://dbpedia.org/ontology/Agent"
      ],
      "image":{
        "full":"https://commons.wikimedia.org/wiki/Special:FilePath/Francesco_Melzi_-_Portrait_of_Leonardo.png",
        "thumbnail":"https://commons.wikimedia.org/wiki/Special:FilePath/Francesco_Melzi_-_Portrait_of_Leonardo.png?width=300"
      },
      "lod":{
        "wikipedia":"https://en.wikipedia.org/wiki/Leonardo_da_Vinci"
      },
      "extras":[
        {
          "start":61,
          "end":69,
          "spot":"Leonardo",
          "confidence":0.35,
          "id":"Q2155112",
          "title":"Bartolomé Leonardo de Argensola",
          "uri":"https://en.wikipedia.org/wiki/Bartolomé_Leonardo_de_Argensola",
          "abstract":"Bartolomé Leonardo de Argensola was baptized at Barbastro on August 26, 1562. He studied at Huesca, took orders, and was presented to the rectory of Villahermosa in 1588. He was attached to the suite of the count de Lemos, viceroy of Naples, in 1610, and succeeded his brother Lupercio as historiographer of Aragon in 1613. He died at Saragossa on February 4, 1631.",
          "label":"Bartolomé Leonardo de Argensola",
          "categories":[
            "1562 births",
            "1631 deaths",
            "People from Barbastro",
            "Spanish poets",
            "Spanish historians",
            "Spanish male poets",
            "University of Salamanca alumni"
          ],
          "tags":[
            "PER",
            "WRITER",
            "BUSINESS"
          ],
          "types":[
            "https://dbpedia.org/ontology/Agent"
          ],
          "image":{
            "full":"https://commons.wikimedia.org/wiki/Special:FilePath/Bartolomé_Leonardo_de_Argensola_(Diputación_Provincial_de_Zaragoza).jpg",
            "thumbnail":"https://commons.wikimedia.org/wiki/Special:FilePath/Bartolomé_Leonardo_de_Argensola_(Diputación_Provincial_de_Zaragoza).jpg?width=300"
          },
          "lod":{
            "wikipedia":"https://en.wikipedia.org/wiki/Bartolomé_Leonardo_de_Argensola"
          }
        },
        {
          "start":61,
          "end":69,
          "spot":"Leonardo",
          "confidence":0.34,
          "id":"Q2107506",
          "title":"Lupercio Leonardo de Argensola",
          "uri":"https://en.wikipedia.org/wiki/Lupercio_Leonardo_de_Argensola",
          "abstract":"Lupercio Leonardo de Argensola (baptised 14 December 1559 – 2 March 1613) was a Spanish dramatist and poet.",
          "label":"Lupercio Leonardo de Argensola",
          "categories":[
            "1559 births",
            "1613 deaths",
            "People from Barbastro",
            "Spanish poets",
            "Spanish dramatists and playwrights",
            "Spanish male dramatists and playwrights",
            "Spanish male poets",
            "University of Zaragoza alumni"
          ],
          "tags":[
            "PER",
            "WRITER",
            "BUSINESS"
          ],
          "types":[
            "https://dbpedia.org/ontology/Agent"
          ],
          "image":{
            "full":"https://commons.wikimedia.org/wiki/Special:FilePath/Lupercio_Leonardo_de_Argensola.jpg",
            "thumbnail":"https://commons.wikimedia.org/wiki/Special:FilePath/Lupercio_Leonardo_de_Argensola.jpg?width=300"
          },
          "lod":{
            "wikipedia":"https://en.wikipedia.org/wiki/Lupercio_Leonardo_de_Argensola"
          }
        },
        {
          "start":61,
          "end":69,
          "spot":"Leonardo",
          "confidence":0.32,
          "id":"Q314488",
          "title":"Leonardo Bruni",
          "uri":"https://en.wikipedia.org/wiki/Leonardo_Bruni",
          "abstract":"Leonardo Bruni (or Leonardo Aretino; c. 1370 – March 9, 1444) was an Italian humanist, historian and statesman, often recognized as the most important humanist historian of the early Renaissance. He has been called the first modern historian. He was the earliest person to write using the three-period view of history: Antiquity, Middle Ages, and Modern. The dates Bruni used to define the periods are not exactly what modern historians use today, but he laid the conceptual groundwork for a tripartite division of history.",
          "label":"Leonardo Bruni",
          "categories":[
            "1370 births",
            "1444 deaths",
            "People from Arezzo",
            "Italian Renaissance writers",
            "15th-century Latin writers",
            "Italian Renaissance humanists",
            "Greek–Latin translators"
          ],
          "tags":[
            "PER",
            "WRITER",
            "BUSINESS"
          ],
          "types":[
            "https://dbpedia.org/ontology/Agent"
          ],
          "image":{
            "full":"https://commons.wikimedia.org/wiki/Special:FilePath/Leonardo_Bruni_2.jpg",
            "thumbnail":"https://commons.wikimedia.org/wiki/Special:FilePath/Leonardo_Bruni_2.jpg?width=300"
          },
          "lod":{
            "wikipedia":"https://en.wikipedia.org/wiki/Leonardo_Bruni"
          }
        },
        {
          "start":61,
          "end":69,
          "spot":"Leonardo",
          "confidence":0.29,
          "id":"Q1271370",
          "title":"Leonardo (footballer, born 1988)",
          "uri":"https://en.wikipedia.org/wiki/Leonardo_(footballer,_born_1988)",
          "abstract":"José Leonardo Ribeiro da Silva (born February 8, 1988), commonly known as Leonardo, is a Brazilian footballer who is currently a free agent and plays as a defender.",
          "label":"Leonardo (footballer, born 1988)",
          "categories":[
            "1989 births",
            "Living people",
            "Brazilian expatriate footballers",
            "Brazilian expatriate sportspeople in the United States",
            "Brazilian footballers",
            "São Paulo FC players",
            "LA Galaxy players",
            "LA Galaxy II players",
            "Houston Dynamo FC players",
            "Orange County SC players"
          ],
          "tags":[
            "PER",
            "WRITER",
            "BUSINESS"
          ],
          "types":[
            "https://dbpedia.org/ontology/Agent"
          ],
          "image":{
            "full":"https://commons.wikimedia.org/wiki/Special:FilePath/HOUSTON_DYNAMO_VS._LA_GALAXY_3_-_0_11.jpg",
            "thumbnail":"https://commons.wikimedia.org/wiki/Special:FilePath/HOUSTON_DYNAMO_VS._LA_GALAXY_3_-_0_11.jpg?width=300"
          },
          "lod":{
            "wikipedia":"https://en.wikipedia.org/wiki/Leonardo_(footballer,_born_1988)"
          }
        }
      ]
    },
    {
      "start":84,
      "end":94,
      "spot":"the Louvre",
      "confidence":1.0,
      "id":"Q19675",
      "title":"Louvre",
      "uri":"https://en.wikipedia.org/wiki/Louvre",
      "abstract":"The Louvre or the Louvre Museum is the worlds most-visited museum, and a historic landmark in Paris, France. It is the home of some of the best-known works of art, including the Mona Lisa and the Venus de Milo. A central landmark of the city, it is located on the Right Bank of the Seine in the citys 1st arrondissement (district or ward). At any given point in time, approximately 38,000 objects from prehistory to the 21st century are being exhibited over an area of 72,735 square meters (782,910 square feet). Attendance in 2021 was 2.8 million due to the COVID-19 pandemic. The museum was closed for 150 days in 2020, and attendance plunged by 72 percent to 2.7 million. Nonetheless, the Louvre still topped the list of most-visited art museums in the world in 2021.",
      "label":"Louvre",
      "categories":[
        "Louvre",
        "1793 establishments in France",
        "Archaeological museums in France",
        "Art museums and galleries in Paris",
        "Art museums established in 1793",
        "Egyptological collections in France",
        "History museums in France",
        "Institut de France",
        "Louvre Palace",
        "Museums in Paris"
      ],
      "tags":[
        "FAC",
        "SPORTS_VENUE",
        "ROAD"
      ],
      "types":[
        
      ],
      "image":{
        "full":"https://commons.wikimedia.org/wiki/Special:FilePath/Louvre_2007_02_24_c.jpg",
        "thumbnail":"https://commons.wikimedia.org/wiki/Special:FilePath/Louvre_2007_02_24_c.jpg?width=300"
      },
      "lod":{
        "wikipedia":"https://en.wikipedia.org/wiki/Louvre"
      },
      "extras":[
        {
          "start":84,
          "end":94,
          "spot":"the Louvre",
          "confidence":0.44,
          "id":"Q3176133",
          "title":"Louvre Abu Dhabi",
          "uri":"https://en.wikipedia.org/wiki/Louvre_Abu_Dhabi",
          "abstract":"The Louvre Abu Dhabi is an art museum located on Saadiyat Island in Abu Dhabi, United Arab Emirates. It runs under an agreement between the UAE and France, signed in March 2007, that allows it to use the Louvres name until 2037, and has been described by the Louvre as \"France’s largest cultural project abroad.\" It is approximately in size, with of galleries, making it the largest art museum in the Arabian peninsula. Artworks from around the world are showcased at the museum, with stated intent to bridge the gap between Eastern and Western art.",
          "label":"Louvre Abu Dhabi",
          "categories":[
            "Louvre Abu Dhabi",
            "2017 establishments in the United Arab Emirates",
            "Art museums established in 2017",
            "Art museums and galleries in the United Arab Emirates",
            "Expressionist architecture",
            "France–United Arab Emirates relations",
            "Jean Nouvel buildings",
            "Saadiyat Island",
            "Arab art scene",
            "Museums established in 2017"
          ],
          "tags":[
            "FAC",
            "SPORTS_VENUE",
            "ROAD"
          ],
          "types":[
            
          ],
          "image":{
            "full":"https://commons.wikimedia.org/wiki/Special:FilePath/Inauguration_du_Louvre_Abou_Dhabi_par_Groupe_F_02.jpg",
            "thumbnail":"https://commons.wikimedia.org/wiki/Special:FilePath/Inauguration_du_Louvre_Abou_Dhabi_par_Groupe_F_02.jpg?width=300"
          },
          "lod":{
            "wikipedia":"https://en.wikipedia.org/wiki/Louvre_Abu_Dhabi"
          }
        },
        {
          "start":84,
          "end":94,
          "spot":"the Louvre",
          "confidence":0.43,
          "id":"Q3390637",
          "title":"Place du Louvre",
          "uri":"https://en.wikipedia.org/wiki/Place_du_Louvre",
          "abstract":"The Place du Louvre is a square immediately to the east of the Palais du Louvre in Paris, France. To the south is the Quai du Louvre and beyond that is the River Seine. The Hôtel du Louvre is also located here, between the Louvre Palace and the Palais Royal.",
          "label":"Place du Louvre",
          "categories":[
            "Squares in Paris|Louvre",
            "Art gallery districts",
            "Louvre Palace",
            "Buildings and structures in the 1st arrondissement of Paris",
            "1850s establishments in France"
          ],
          "tags":[
            "FAC",
            "SPORTS_VENUE",
            "ROAD"
          ],
          "types":[
            
          ],
          "image":{
            "full":"https://commons.wikimedia.org/wiki/Special:FilePath/P1070100_Paris_Ier_place_du_Louvre_rwk.jpg",
            "thumbnail":"https://commons.wikimedia.org/wiki/Special:FilePath/P1070100_Paris_Ier_place_du_Louvre_rwk.jpg?width=300"
          },
          "lod":{
            "wikipedia":"https://en.wikipedia.org/wiki/Place_du_Louvre"
          }
        },
        {
          "start":84,
          "end":94,
          "spot":"the Louvre",
          "confidence":0.42,
          "id":"Q15992417",
          "title":"Louvre Castle",
          "uri":"https://en.wikipedia.org/wiki/Louvre_Castle",
          "abstract":"The Louvre Castle, also known as the Medieval Louvre, was a castle built by King Philip II of France on the right bank of the Seine, to reinforce the city wall he had built around Paris. It was demolished in stages between 1528 and 1660 to make way for the expanded Louvre Palace. ",
          "label":"Louvre Castle",
          "categories":[
            "Louvre Palace",
            "1202 establishments in Europe",
            "1200s establishments in France",
            "Châteaux in Paris|Louvre Castle",
            "17th-century disestablishments in France",
            "Buildings and structures in the 1st arrondissement of Paris",
            "Buildings and structures completed in 1202",
            "Houses completed in the 13th century",
            "Buildings and structures demolished in the 17th century",
            "Demolished buildings and structures in Paris"
          ],
          "tags":[
            "FAC",
            "SPORTS_VENUE",
            "ROAD"
          ],
          "types":[
            
          ],
          "image":{
            "full":"https://commons.wikimedia.org/wiki/Special:FilePath/Louvre_-_Les_Très_Riches_Heures.jpg",
            "thumbnail":"https://commons.wikimedia.org/wiki/Special:FilePath/Louvre_-_Les_Très_Riches_Heures.jpg?width=300"
          },
          "lod":{
            "wikipedia":"https://en.wikipedia.org/wiki/Louvre_Castle"
          }
        },
        {
          "start":84,
          "end":94,
          "spot":"the Louvre",
          "confidence":0.41,
          "id":"Q405543",
          "title":"Louvre Lens",
          "uri":"https://en.wikipedia.org/wiki/Louvre_Lens",
          "abstract":"The Louvre-Lens is an art museum located in Lens, France, approximately 200 kilometers north of Paris. It displays objects from the collections of the Musée du Louvre that are lent to the gallery on a medium- or long-term basis. The Louvre-Lens annex is part of an effort to provide access to French cultural institutions for people who live outside of Paris. Though the museum maintains close institutional links with the Louvre, it is primarily funded by the Nord-Pas-de-Calais region.",
          "label":"Louvre Lens",
          "categories":[
            "Lens, Pas-de-Calais",
            "Louvre",
            "Art museums and galleries in France",
            "Museums in Pas-de-Calais",
            "2012 establishments in France",
            "Art museums established in 2012",
            "SANAA buildings"
          ],
          "tags":[
            "FAC",
            "SPORTS_VENUE",
            "ROAD"
          ],
          "types":[
            
          ],
          "image":{
            "full":"https://commons.wikimedia.org/wiki/Special:FilePath/0_La_Liberté_guidant_le_peuple_-_Eugène_Delacroix_(1).JPG",
            "thumbnail":"https://commons.wikimedia.org/wiki/Special:FilePath/0_La_Liberté_guidant_le_peuple_-_Eugène_Delacroix_(1).JPG?width=300"
          },
          "lod":{
            "wikipedia":"https://en.wikipedia.org/wiki/Louvre_Lens"
          }
        }
      ]
    },
    {
      "start":98,
      "end":103,
      "spot":"Paris",
      "confidence":1.0,
      "id":"Q90",
      "title":"Paris",
      "uri":"https://en.wikipedia.org/wiki/Paris",
      "abstract":"Paris is the capital and most populous city of France, with an estimated population of 2,165,423 residents in 2019 in an area of more than 105 km² (41 sq mi), making it the 34th most densely populated city in the world in 2020. Since the 17th century, Paris has been one of the worlds major centres of finance, diplomacy, commerce, fashion, gastronomy, science, and arts, and has sometimes been referred to as the capital of the world. The City of Paris is the centre and seat of government of the region and province of Île-de-France, or Paris Region, with an estimated population of 12,997,058 in 2020, or about 18% of the population of France, making it in 2020 the second largest metropolitan area in the OECD, and 14th largest in the world in 2015. The Paris Region had a GDP of €709 billion ($808 billion) in 2017. According to the Economist Intelligence Unit Worldwide Cost of Living Survey, in 2021 Paris was the city with the second-highest cost of living in the world, tied with Singapore, and after Tel Aviv.",
      "label":"Paris",
      "categories":[
        "Paris",
        "3rd-century BC establishments",
        "Capitals in Europe",
        "Catholic pilgrimage sites",
        "Cities in France",
        "Cities in Île-de-France",
        "Companions of the Liberation",
        "Departments of Île-de-France",
        "European culture",
        "French culture"
      ],
      "tags":[
        "CITY",
        "LOC",
        "COUNTRY"
      ],
      "types":[
        "https://dbpedia.org/ontology/Place"
      ],
      "image":{
        "full":"https://commons.wikimedia.org/wiki/Special:FilePath/Paris_-_Eiffelturm_und_Marsfeld2.jpg",
        "thumbnail":"https://commons.wikimedia.org/wiki/Special:FilePath/Paris_-_Eiffelturm_und_Marsfeld2.jpg?width=300"
      },
      "lod":{
        "wikipedia":"https://en.wikipedia.org/wiki/Paris"
      },
      "extras":[
        {
          "start":98,
          "end":103,
          "spot":"Paris",
          "confidence":0.81,
          "id":"Q2835080",
          "title":"Parys",
          "uri":"https://en.wikipedia.org/wiki/Parys",
          "abstract":"Parys (pronounced ) is a town situated on the banks of the Vaal River in the Free State province of South Africa. The name of the town is the Afrikaans translation of Paris. The name was given by a German surveyor named Schilbach who had participated in the siege of Paris during the Franco-Prussian War and the location next to the Vaal reminded him of Paris on the River Seine. The area of Parys also includes the two townships Tumahole and Schonkenville. ",
          "label":"Parys",
          "categories":[
            "Populated places in the Ngwathe Local Municipality",
            "Populated places established in 1882"
          ],
          "tags":[
            "CITY",
            "LOC",
            "COUNTRY"
          ],
          "types":[
            "https://dbpedia.org/ontology/Place"
          ],
          "image":{
            "full":"https://commons.wikimedia.org/wiki/Special:FilePath/Nederduitse_Gereformeerde_Mother_Church_Parys-006.jpg",
            "thumbnail":"https://commons.wikimedia.org/wiki/Special:FilePath/Nederduitse_Gereformeerde_Mother_Church_Parys-006.jpg?width=300"
          },
          "lod":{
            "wikipedia":"https://en.wikipedia.org/wiki/Parys"
          }
        },
        {
          "start":98,
          "end":103,
          "spot":"Paris",
          "confidence":0.8,
          "id":"Q19660",
          "title":"Bucharest",
          "uri":"https://en.wikipedia.org/wiki/Bucharest",
          "abstract":"Bucharest is the capital and largest city of Romania, as well as its cultural, industrial, and financial centre. It is located in the southeast of the country, on the banks of the Dâmbovița River, less than north of the Danube River and the Bulgarian border.",
          "label":"Bucharest",
          "categories":[
            "Bucharest",
            "Capitals in Europe",
            "Cities in Romania",
            "Capitals of Romanian counties",
            "Localities in Muntenia",
            "Market towns in Wallachia",
            "Holocaust locations in Romania",
            "1459 establishments in the Ottoman Empire",
            "Populated places established in the 1450s",
            "1968 establishments in Romania"
          ],
          "tags":[
            "CITY",
            "LOC",
            "COUNTRY"
          ],
          "types":[
            "https://dbpedia.org/ontology/Place"
          ],
          "image":{
            "full":"https://commons.wikimedia.org/wiki/Special:FilePath/Bucharest-Skyline-01.jpg",
            "thumbnail":"https://commons.wikimedia.org/wiki/Special:FilePath/Bucharest-Skyline-01.jpg?width=300"
          },
          "lod":{
            "wikipedia":"https://en.wikipedia.org/wiki/Bucharest"
          }
        },
        {
          "start":98,
          "end":103,
          "spot":"Paris",
          "confidence":0.6,
          "id":"Q3181341",
          "title":"Paris, Kentucky",
          "uri":"https://en.wikipedia.org/wiki/Paris,_Kentucky",
          "abstract":"Paris is a home rule-class city in Bourbon County, Kentucky. It lies northeast of Lexington on the Stoner Fork of the Licking River. Paris is the seat of its county and forms part of the Lexington–Fayette Metropolitan Statistical Area. As of 2020 it has a population of 9,846. ",
          "label":"Paris, Kentucky",
          "categories":[
            "Cities in Bourbon County, Kentucky",
            "Cities in Kentucky",
            "County seats in Kentucky",
            "Lexington–Fayette metropolitan area",
            "Populated places established in 1789",
            "1789 establishments in Virginia"
          ],
          "tags":[
            "CITY",
            "LOC",
            "COUNTRY"
          ],
          "types":[
            "https://dbpedia.org/ontology/Place"
          ],
          "image":{
            "full":"https://commons.wikimedia.org/wiki/Special:FilePath/Parisstreet.JPG",
            "thumbnail":"https://commons.wikimedia.org/wiki/Special:FilePath/Parisstreet.JPG?width=300"
          },
          "lod":{
            "wikipedia":"https://en.wikipedia.org/wiki/Paris,_Kentucky"
          }
        },
        {
          "start":98,
          "end":103,
          "spot":"Paris",
          "confidence":0.55,
          "id":"Q830149",
          "title":"Paris, Texas",
          "uri":"https://en.wikipedia.org/wiki/Paris,_Texas",
          "abstract":"Paris is a city and county seat of Lamar County, Texas, United States. As of the 2010 census, the population of the city was 25,171. Paris is in Northeast Texas at the western edge of the Piney Woods.",
          "label":"Paris, Texas",
          "categories":[
            "Paris, Texas",
            "Cities in Texas",
            "Cities in Lamar County, Texas",
            "County seats in Texas",
            "Micropolitan areas of Texas",
            "1844 establishments in the Republic of Texas",
            "Populated places established in 1844"
          ],
          "tags":[
            "CITY",
            "LOC",
            "COUNTRY"
          ],
          "types":[
            "https://dbpedia.org/ontology/Place"
          ],
          "image":{
            "full":"https://commons.wikimedia.org/wiki/Special:FilePath/Main_Street_at_The_Plaza_Paris_Texas_DSC_0620_ad.jpg",
            "thumbnail":"https://commons.wikimedia.org/wiki/Special:FilePath/Main_Street_at_The_Plaza_Paris_Texas_DSC_0620_ad.jpg?width=300"
          },
          "lod":{
            "wikipedia":"https://en.wikipedia.org/wiki/Paris,_Texas"
          }
        }
      ]
    }
  ],
  "lang":"en",
  "timestamp":"2022-05-23T17:28:04.955988"
}
```

</details>
