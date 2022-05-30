WIKIPEDIA_PAGE_URI_PREFIX = "https://en.wikipedia.org/wiki"
WIKIPEDIA_FILE_URI_PREFIX = "https://commons.wikimedia.org/wiki/Special:FilePath"
ONTOLOGY_URI_PREFIX = "https://dbpedia.org/ontology"
TAG_TO_TYPE_MAP = {
    "per": "Person",
    "loc": "Place",
    "org": "Organisation",
    "business": "Company"
}
TAG_TO_TYPE_LIST_MAP = {
    "per": [
        "Agent",
        "Person"
    ],
    "loc": [
        "Place"
    ],
    "org": [
        "Agent",
        "Organisation"
    ],
    "business": [
        "Agent",
        "Organisation",
        "Company"
    ]
}
