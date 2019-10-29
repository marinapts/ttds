ranked_docs_for_queries = dict()

with open('./systems/S1.results', 'r') as f:
    lines = f.readlines()
    system_results_dict = dict()

    for line in lines:
        query_id, _, doc_id, rank_of_doc, doc_score, _ = line.split(' ')
        doc_details = {'rank': rank_of_doc, 'score': doc_score}

        if query_id in system_results_dict:
            system_results_dict[query_id].update({doc_id: doc_details})
        else:
            system_results_dict.setdefault(query_id, {doc_id: doc_details})
