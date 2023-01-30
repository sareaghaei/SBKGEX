from SPARQLWrapper import SPARQLWrapper, JSON
import pandas as pd


input_data = '../data/QALD/input_data.csv'
sparql_endpoint = "http://dbpedia.org/sparql"  # replace with local end point
filtered_properties = ['<http://dbpedia.org/ontology/wikiPageWikiLink>']  # replace with the unnecessary relations
filtered_properties = ', '.join(filtered_properties)
output_dir = "neighborhood_files/neighborhood_question_"


def fetch_1hop_neighbors(entity, filtered_properties, sparql_endpoint, file, entity_set):
    f_out = open(file, 'w')
    sparql = SPARQLWrapper(sparql_endpoint)
    sparql.setQuery("""
    SELECT distinct ?o ?p
    WHERE { """ + entity + """ ?p ?o filter (?p not in (""" + filtered_properties + """))}"""
                    )
    sparql.setReturnFormat(JSON)
    results = sparql.query().convert()
    for r in results["results"]["bindings"]:
        p = "<" + r["p"]["value"] + ">"
        o = "<" + r["o"]["value"] + ">"
        fact = entity + "\t" + p + "\t" + o + "\n"
        f_out.write(fact)
        if o not in entity_set:
            entity_set.add(o)

    sparql.setQuery("""
        SELECT distinct ?s ?p
        WHERE { ?s ?p """ + entity + """ filter (?p not in (""" + filtered_properties + """))}"""
                    )
    sparql.setReturnFormat(JSON)
    results = sparql.query().convert()
    for r in results["results"]["bindings"]:
        p = "<" + r["p"]["value"] + ">"
        s = "<" + r["s"]["value"] + ">"
        fact = s + "\t" + p + "\t" + entity + "\n"
        f_out.write(fact)
        if s not in entity_set:
            entity_set.add(s)
    entity_set.remove(entity)
    f_out.close()
    return entity_set


def fetch_2hop_neighbors(entity, filtered_properties, sparql_endpoint, file, entity_set):
    en_set = fetch_1hop_neighbors(entity, filtered_properties, sparql_endpoint, file, entity_set)
    entity_set = en_set
    for e in list(en_set):
        if e.startswith('<http'):
            eset = fetch_1hop_neighbors(e, filtered_properties, sparql_endpoint, file, entity_set)
            entity_set.update(eset)
    return entity_set


def fetch_3hop_neighbors(entity, filtered_properties, sparql_endpoint, file, entity_set):
    en_set = fetch_2hop_neighbors(entity, filtered_properties, sparql_endpoint, file, entity_set)
    entity_set = en_set
    for e in en_set:
        eset = fetch_1hop_neighbors(e, filtered_properties, sparql_endpoint, file, entity_set)
        entity_set.update(eset)
    return entity_set


def fetch_4hop_neighbors(entity, filtered_properties, sparql_endpoint, file, entity_set):
    en_set = fetch_3hop_neighbors(entity, filtered_properties, sparql_endpoint, file, entity_set)
    entity_set = en_set
    for e in en_set:
        eset = fetch_1hop_neighbors(e, filtered_properties, sparql_endpoint, file, entity_set)
        entity_set.update(eset)
    return entity_set


def fetch_neighbors(sparql_endpoint, topic_entity, hop, neighbor_file):
    entity_set = {topic_entity}
    if hop == 1:
        fetch_1hop_neighbors(topic_entity, filtered_properties, sparql_endpoint, neighbor_file, entity_set)
    if hop == 2:
        fetch_2hop_neighbors(topic_entity, filtered_properties, sparql_endpoint, neighbor_file, entity_set)
    if hop == 3:
        fetch_3hop_neighbors(topic_entity, filtered_properties, sparql_endpoint, neighbor_file, entity_set)
    if hop == 4:
        fetch_4hop_neighbors(topic_entity, filtered_properties, sparql_endpoint, neighbor_file, entity_set)


def main():
    df = pd.read_csv(input_data)
    for index, row in df.iterrows():
        neighbor_file = output_dir + str(row['id']) + ".nxhd"
        fetch_neighbors(sparql_endpoint, row['topic_entity'], row['hop'], neighbor_file)


# This part makes a neighborhood graph by extracting n-hop neighbors around a topic entity
if __name__ == "__main__":
    main()
