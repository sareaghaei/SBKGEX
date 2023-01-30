import os
import json
import random
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.preprocessing import normalize
import math

random.seed(0)

MAX_FACTS = 5000000
MAX_ITER = 20
RESTART = 0.8

NOTFOUNDSCORE = 0.
EXPONENT = 2.
MAX_SEEDS = 1
DECOMPOSE_PPV = True
SEED_WEIGHTING = True
RELATION_WEIGHTING = True
FOLLOW_NONCVT = True
USEANSWER = False

question_answer_json = "../data/QALD/qald-6-test-multilingual_edit.json"
question_subgraph_dir = "./knowledge_subgraphs/question_"
neighborhood_dir = "../neighborhoodRetrieval/neighborhood_files/"
input_data = '../data/QALD/input_data.csv'

# it is for FreeBase benchmark QA-datasets
def _filter_relation(relation):
    if relation == "<fb:common.topic.notable_types>": return False
    domain = relation[4:-1].split(".")[0]
    if domain == "type" or domain == "common": return True
    return False


def _read_facts(fact_file,  # relation_embeddings, question_embedding,
                seeds):
    """Read all triples from the fact file and create a sparse adjacency
    matrix between the entities. Returns mapping of entities to their
    indices, a mapping of relations to the
    and the combined adjacency matrix."""
    seeds_found = set()
    with open(fact_file) as f:
        entity_map = {}
        relation_map = {}
        all_row_ones, all_col_ones = [], []
        num_entities = 0
        num_facts = 0
        for line in f:
            try:
                e1, rel, e2 = line.strip().split(None, 2)
            except ValueError:
                continue
            if _filter_relation(rel): continue
            if e1 not in entity_map:
                entity_map[e1] = num_entities
                num_entities += 1
            if e2 not in entity_map:
                entity_map[e2] = num_entities
                num_entities += 1
            if rel not in relation_map:
                relation_map[rel] = [[], []]
            if e1 in seeds: seeds_found.add(e1)
            if e2 in seeds: seeds_found.add(e2)
            all_row_ones.append(entity_map[e1])
            all_col_ones.append(entity_map[e2])
            all_row_ones.append(entity_map[e2])
            all_col_ones.append(entity_map[e1])
            relation_map[rel][0].append(entity_map[e1])
            relation_map[rel][1].append(entity_map[e2])
            num_facts += 1
            if num_facts == MAX_FACTS:
                break
    if not relation_map:
        return {}, {}, None
    for rel in relation_map:
        row_ones, col_ones = relation_map[rel]
        m = csr_matrix(
            (np.ones((len(row_ones),)), (np.array(row_ones), np.array(col_ones))),
            shape=(num_entities, num_entities))
        relation_map[rel] = normalize(m, norm="l1", axis=1)
        if RELATION_WEIGHTING:
            score = 1
            relation_map[rel] = relation_map[rel] * np.power(score, EXPONENT)
    if DECOMPOSE_PPV:
        adj_mat = sum(relation_map.values()) / len(relation_map)
    else:
        adj_mat = csr_matrix(
            (np.ones((len(all_row_ones),)), (np.array(all_row_ones), np.array(all_col_ones))),
            shape=(num_entities, num_entities))
    return entity_map, relation_map, normalize(adj_mat, norm="l1", axis=1)


def _personalized_pagerank(seed, W):
    """Return the PPR vector for the given seed and adjacency matrix.
    Args:
        seed: A sparse matrix of size E x 1.
        W: A sparse matrix of size E x E whose rows sum to one.
    Returns:
        ppr: A vector of size E.
    """
    restart_prob = RESTART
    r = restart_prob * seed
    s_ovr = np.copy(r)
    for i in range(MAX_ITER):
        r_new = 0.8 * (W.transpose().dot(r))
        s_ovr = s_ovr + r_new
        delta = abs(r_new.sum())
        if delta < 1e-5: break
        r = r_new
    return np.squeeze(s_ovr)

def bidppr_pagerank(seed, W, restart_prob=0.8, max_iter=20):
    r = restart_prob * seed
    s = restart_prob * seed
    s_ovr = np.copy(r)

    for i in range(max_iter):
        r_new = (1. - restart_prob) * (W.transpose().dot(r))
        s_new = (1. - restart_prob) * (W.dot(s))

        sr_new = np.square(r_new)
        norm = np.sum(sr_new)
        norm = math.sqrt(norm)
        r_new = r_new * 1 / norm

        ss_new = np.square(s_new)
        norm = np.sum(ss_new)
        norm = math.sqrt(norm)
        s_new = s_new * 1 / norm

        s_ovr = s_ovr + r_new + 0.01 * s_new

        delta = abs(r_new.sum()) + abs(s_new.sum())

        if delta < 1e-5: break
        r = s_ovr
        s = s_ovr
    return np.squeeze(s_ovr)


def _get_subgraph(entities, kb_r, multigraph_W, MAX_ENT, ppr_type, entity_map, file):
    if ppr_type == 'PPR':
        seed = np.zeros((multigraph_W.shape[0], 1))
    elif ppr_type == 'BiPPR':
        seed = np.ones((multigraph_W.shape[0], 1))

    if not SEED_WEIGHTING:
        seed[entities] = 1. / len(set(entities))
    else:

        seed[entities] = np.expand_dims(np.arange(len(entities), 0, -1),
                                        axis=1)
        seed = seed / seed.sum()

        if ppr_type == 'BiPPR':
            seed[entities] = seed[entities] + 1
    ppr = bidppr_pagerank(seed, multigraph_W)
    sorted_idx = np.argsort(ppr)[::-1]
    extracted_ents = sorted_idx[:MAX_ENT]
    extracted_scores = ppr[sorted_idx[:MAX_ENT]]
    # check if any ppr values are nearly zero
    zero_idx = np.where(ppr[extracted_ents] < 1e-6)[0]
    if zero_idx.shape[0] > 0:
        extracted_ents = extracted_ents[:zero_idx[0]]
    extracted_tuples = []
    ents_in_tups = set()
    f_out = open(file, 'w')
    for relation in kb_r:
        submat = kb_r[relation][extracted_ents, :]
        submat = submat[:, extracted_ents]
        row_idx, col_idx = submat.nonzero()
        for ii in range(row_idx.shape[0]):
            extracted_tuples.append(
                (extracted_ents[row_idx[ii]], relation,
                 extracted_ents[col_idx[ii]]))
            ents_in_tups.add((extracted_ents[row_idx[ii]],
                              extracted_scores[row_idx[ii]]))
            ents_in_tups.add((extracted_ents[col_idx[ii]],
                              extracted_scores[col_idx[ii]]))
            s = list(filter(lambda x: entity_map[x] == extracted_ents[row_idx[ii]], entity_map))[0]
            o = list(filter(lambda x: entity_map[x] == extracted_ents[col_idx[ii]], entity_map))[0]
            fact = s + "\t" + relation + "\t" + o + "\n"
            f_out.write(fact)
    f_out.close()
    return extracted_tuples, list(ents_in_tups)


def _read_seeds():
    """Return map from question ids to seed entities."""
    seed_map = {}
    df = pd.read_csv(input_data)
    for index, row in df.iterrows():
        qid = row["id"]
        arr = []
        if row['topic_entity_count'] > 1:
            main_entity = row['topic_entity'].split(',')
            for e in main_entity:
                arr.append(e)
        else:
            arr.append(row["topic_entity"])

        seed_map[qid] = arr
    return seed_map


def _convert_to_readable(tuples, inv_map):
    readable_tuples = []
    for tup in tuples:
        readable_tuples.append([
            {"kb_id": inv_map[tup[0]], "text": inv_map[tup[0]]},
            {"rel_id": tup[1], "text": tup[1]},
            {"kb_id": inv_map[tup[2]], "text": inv_map[tup[2]]},
        ])
    return readable_tuples


def _readable_entities(entities, inv_map):
    readable_entities = []
    try:
        for ent, sc in entities:
            readable_entities.append(
                {"text": inv_map[ent], "kb_id": inv_map[ent],
                 "pagerank_score": sc})
    except TypeError:
        for ent in entities:
            readable_entities.append(
                {"text": inv_map[ent], "kb_id": inv_map[ent]})
    return readable_entities


def _get_answer_coverage(answers, entities, inv_map):
    found, total = 0., 0
    all_entities = set([inv_map[ee] for ee, _ in entities])
    for answer in answers:

        if "<" + answer + ">" in all_entities:
            found += 1.
        total += 1
    return found / total


if __name__ == "__main__":

    questions = json.load(open(question_answer_json, encoding='utf-8-sig'))
    seed_map = _read_seeds()

    num_ent = [500]  # , 1000, 1500, 2000
    result = []
    ppr_type = 'PPR'  # 'BiPPR'
    for MAX_ENT in num_ent:
        answer_recall, total = 0.0, 0
        max_recall = 0.
        bad_questions = []
        num_empty = 0
        for q in range(1, 101):

            fact_file = os.path.join(
                neighborhood_dir, "neighborhood_question_" + str(q) + ".nxhd")

            if not os.path.exists(fact_file):
                print("fact file not found for %s" % q)
                entity_map, relation_map, adj_mat = {}, {}, None
                continue
            else:
                entity_map, relation_map, adj_mat = _read_facts(fact_file, seed_map[q])
            inv_map = {i: k for k, i in entity_map.items()}
            seed_entities = []
            ans_entities = []
            if q in seed_map:
                for e in seed_map[q]:
                    if e in entity_map:
                        seed_entities.append(entity_map[e])

                answ = []
                for i, a in enumerate(questions["questions"][q - 1]["answers"][0]["results"]["bindings"]):
                    if "uri" in a:
                        answ.append(a["uri"]["value"])
                    if "s" in a:
                        answ.append(a["s"]["value"])
                    if "d" in a:
                        answ.append(a["d"]["value"])
                    if "n" in a:
                        answ.append(a["n"]["value"])

                for a in answ:
                    if a in entity_map:
                        ans_entities.append(entity_map[a])
            if not seed_entities:
                print("No seeds found for %s!" % q)
                extracted_tuples, extracted_ents = [], []
            elif adj_mat is None:
                print("No facts for %s!" % q)
                extracted_tuples, extracted_ents = [], []
            else:
                sd = seed_entities + ans_entities if USEANSWER else seed_entities
                q_file = neighbor_file = question_subgraph_dir + str(q) + ".nxhd"
                extracted_tuples, extracted_ents = _get_subgraph(
                    sd, relation_map, adj_mat, MAX_ENT, ppr_type, entity_map, q_file)

            if not extracted_tuples:
                num_empty += 1
            if not answ:
                curr_recall = 0.
                cmax_recall = 0.
            else:
                curr_recall = _get_answer_coverage(answ,
                                                   extracted_ents, inv_map)

                cmax_recall = float(len([answer for answer in answ
                                         if answer in entity_map])) / len(answ)

            if curr_recall < 1:
                print(str(q) + "         " + str(curr_recall))
            answer_recall += curr_recall
            max_recall += cmax_recall
            total += 1
            data = {

                "id": q,
                "subgraph": {
                    "entities": _readable_entities(extracted_ents, inv_map),
                    "tuples": _convert_to_readable(extracted_tuples, inv_map)
                }
            }

        result.append({"num_ent": MAX_ENT, "dataset": "QALD", "coverage": (answer_recall / total), "Algorithm": "NPPR"})

    print("Answer recall = %.3f" % (answer_recall / total))
    print("Upper Bound = %.3f" % (max_recall / total))
    print("Example questions with low recall: ")
    print("\n".join(["%s\t%s\t%s\t%.2f" % (
        item[0], item[1], ",".join(ss for ss in item[2]), item[3])
                     for item in bad_questions[:10]]))
