## Adding a new dataset

* Add prompts to src/downstream_evaluation/prompts/dataset_prompts
* Update downstream_evaluation_datasets_prompts_dict at src/downstream_evaluation/prompts/prompts.py
* Update get_answer_extraction_template at src/downstream_evaluation/prompts/prompts.py
* Update downstream_evaluation_datasets_list, display_name_of_downstream_evaluation_dataset_dict, and downstream_evaluation_dataset_name_to_category_dict at src/config.py
* Update src/load_dataset/load_existing_dataset.py

* Update src/downstream_evaluation/evaluation/utils/extract_final_answer.py
* Update src/downstream_evaluation/evaluation/utils/compare_final_answer.py
