import math
import os
from collections import Counter, defaultdict
import pickle

import numpy as np
import pandas as pd
import scipy.stats as stats
from pymongo import MongoClient
import networkx as nx
import datetime
from datetime import datetime, timedelta, date
from Build_reverse_identity_dictionary import Build_reverse_identity_dictionary
import hypernetx as hnx


def calculate_1d_entropy(graph):
    def compute_entropy(probabilities):
        return -sum(p * math.log2(p) for p in probabilities if p > 0)

    total_edges = graph.number_of_edges()

    probabilities = {}
    for node in graph.nodes():
        adjacent_edge_sum = graph.degree(node)
        probabilities[node] = adjacent_edge_sum / (2 * total_edges)

    original_entropy = compute_entropy(probabilities.values())

    file_entropy_contributions = {}
    for node in graph.nodes():
        file_entropy_contributions[node] = probabilities[node] * original_entropy

    return original_entropy, file_entropy_contributions


def calculate_change_entropy(commit_set_of_files, g):
    file_change_counts = Counter()

    for files in commit_set_of_files.values():
        for file in files:
            file_change_counts[file] += 1

    total_changes = sum(file_change_counts.values())

    probabilities = {file: count / total_changes for file, count in file_change_counts.items()}
    print(f"probability sum = {sum(probabilities.values())}")

    entropy = -sum(p * math.log2(p) for p in probabilities.values() if p > 0)

    hcpf2_values = {
        file: probabilities[file] * entropy
        for file in file_change_counts if file in g.nodes
    }

    return entropy, hcpf2_values


class Task2:

    def __init__(self):
        self.client = MongoClient("mongodb://localhost:27017/")
        self.db = self.client["smartshark"]
        self.commit_project = self.db["commit_with_project_info"]
        self.commit_project_records = list(self.commit_project.find({}, {}))
        self.commit_file = self.db["file_action"]
        self.file_path = self.db["file"]
        self.commit_file_records = list(self.commit_file.find({}, {"file_id": 1, "commit_id": 1}))
        self.file_path_records = list(self.file_path.find({}, {"_id": 1, "path": 1}))
        self.BRID = Build_reverse_identity_dictionary()
        self.BRID.reading_identity_and_people_and_building_reverse_identity_dictionary()
        self.commit_set_of_files = dict()
        self.commit_set_of_src_files = dict()
        self.commit_project_data = dict()
        self.commit_time_unsort = dict()
        self.commit_time = dict()
        self.version_commit_time = dict()
        self.src_files_set = set()
        self.file_path_data = dict()
        self.graph_data = dict()
        self.g = nx.Graph()
        self.g1 = nx.Graph()
        self.path_tree = nx.DiGraph()
        self.node_mapping = dict()
        self.file_level1 = dict()
        self.g_test = nx.Graph()
        self.file_extension = dict()
        self.hyper_graph = dict()

    def build_commit_set_of_files(self):
        for element in self.commit_file_records:
            file_id = element["file_id"]
            commit_id = element["commit_id"]
            if commit_id not in self.commit_set_of_files.keys():
                self.commit_set_of_files[commit_id] = set()
            self.commit_set_of_files[commit_id].add(file_id)

    def build_commit_set_of_src_files(self, project):
        self.commit_set_of_src_files.clear()
        for commit in self.commit_set_of_files:
            if self.commit_project_data[commit] == project:
                files = self.commit_set_of_files[commit]
                files = list(files)
                r = len(files)
                for i in range(r):
                    path1 = self.file_path_data[files[i]]
                    pos1 = path1.rfind('/')
                    if pos1 != -1:
                        trim1 = path1[0:pos1:1]
                        modules1 = trim1.split("/")
                        if "test" not in modules1:
                            if files[i] in self.src_files_set:
                                if commit not in self.commit_set_of_src_files.keys():
                                    self.commit_set_of_src_files[commit] = list()
                                if files[i] not in self.commit_set_of_src_files[commit]:
                                    self.commit_set_of_src_files[commit].append(files[i])

    def build_file_path(self):
        src_list = ['asm',
                    'c',
                    'class',
                    'cpp',
                    'CPP',
                    'java',
                    'js',
                    'jsp',
                    'pl',
                    'py',
                    'R',
                    'r', 'mhtml',
                    'jspf',
                    'dml',
                    'pydml',
                    'aj',
                    'RexsterExtension',
                    'awk',
                    'MPL']
        for element in self.file_path_records:
            file_id = element["_id"]
            path = element["path"]
            if file_id not in self.file_path_data.keys():
                self.file_path_data[file_id] = path
        for file in self.file_path_data:
            path = self.file_path_data[file]
            pat = path.split(".")
            pos = pat[-1].rfind('/')
            if pos == -1:
                if len(pat) > 1:
                    self.file_extension[file] = pat[-1]
        for file in self.file_extension:
            ex = self.file_extension[file]
            if str(ex) in src_list:
                self.src_files_set.add(file)

    def build_commit_time(self):
        for element in self.commit_project_records:
            commit_id = element["_id"]
            project_name = element["project_name_info"]["name"]
            c_time = element["committer_date"]
            if commit_id not in self.commit_project_data.keys():
                self.commit_project_data[commit_id] = project_name
            if commit_id not in self.commit_time_unsort.keys():
                self.commit_time_unsort[commit_id] = c_time
        self.commit_time = {key: value for key, value in sorted(self.commit_time_unsort.items(),
                                                                key=lambda item: item[1])}

    def build_time(self, project, t1):
        c = 0
        time_list = list()
        for commit in self.commit_time:
            if self.commit_project_data[commit] == project:
                time_list.append(self.commit_time[commit])
        t1min1 = min(time_list)
        t7max = max(time_list)
        print(f"Min time t1min1 = {t1min1}")
        print(f"Max time t7max = {t7max}")
        print(f"normal graph {datetime.now()}")
        self.create_normal_graph(t1, project)

    def create_normal_graph(self, releases, project):
        self.g.clear()
        self.g1.clear()
        self.graph_data.clear()
        self.hyper_graph.clear()
        all_data = []

        # Load project release commits
        with open('project_release_commits_final_new.pkl', 'rb') as f:
            project_release_commits = pickle.load(f)

        for t1 in releases:
            # Clear the graph for each release
            self.g.clear()
            self.g1.clear()
            self.hyper_graph.clear()
            release_commits_files = dict()
            pair_commit_count = defaultdict(int)

            # Load the new pickle files for each release
            process_metrics_path = f'{project}_process_metrics_release_new_entropy_non_fatty.pkl'
            file_bugs_path = f'{project}_{t1}_file_bugs_release_entropy_non_fatty.pkl'
            file_bugs_priority_path = f'{project}_{t1}_file_bugs_priority_release_entropy_non_fatty.pkl'

            if not os.path.exists(process_metrics_path) or not os.path.exists(file_bugs_path) or not os.path.exists(
                    file_bugs_priority_path):
                print(f"Files for release {t1} are missing. Skipping...")
                continue

            with open(process_metrics_path, 'rb') as f:
                process_metrics = pickle.load(f)

            with open(file_bugs_path, 'rb') as f:
                file_bugs = pickle.load(f)

            with open(file_bugs_priority_path, 'rb') as f:
                file_bugs_priority = pickle.load(f)

            # Extract commits for the current release
            for pro in project_release_commits:
                if pro == project:
                    for rel in project_release_commits[pro].keys():
                        if rel == t1:
                            if t1 not in self.graph_data:
                                self.graph_data[t1] = dict()
                            for commit in project_release_commits[pro][rel]:
                                self.graph_data[t1][commit] = self.commit_time[commit]

            # Create graph for the current release
            for commit in self.commit_set_of_src_files:
                if self.commit_project_data[commit] == project:
                    if commit in self.graph_data[t1]:
                        release_commits_files[commit] = self.commit_set_of_src_files[commit]
                        files = self.commit_set_of_src_files[commit]
                        if len(files) <= 30:
                            ordered_pairs = {(x, y) for x in files for y in files if x != y}
                            self.g.add_edges_from(ordered_pairs)
                            if commit not in self.hyper_graph.keys():
                                self.hyper_graph[commit] = files

                            # Count co-commit occurrences for entropy calculation
                            for pair in ordered_pairs:
                                pair_commit_count[pair] += 1

            # Add edges with weights to the graph
            for (file1, file2), weight in pair_commit_count.items():
                self.g1.add_edge(file1, file2, weight=weight)

            H = hnx.Hypergraph(self.hyper_graph)

            # Calculate entropies
            _, file_co_change_entropy = calculate_1d_entropy(self.g)
            _, HCPF2 = calculate_change_entropy(release_commits_files, self.g)

            # Combine all required columns for each file
            release_data = []
            for file in self.g.nodes():
                row = {
                    'File': file,
                    "comm": process_metrics['comm'][t1].get(file, 0),
                    "adev": process_metrics['adev'][t1].get(file, 0),
                    "ddev": process_metrics['ddev'][t1].get(file, 0),
                    "add": process_metrics['add'][t1].get(file, 0),
                    "del": process_metrics['del'][t1].get(file, 0),
                    "own": process_metrics['own'][t1].get(file, 0),
                    "minor": process_metrics['minor'][t1].get(file, 0),
                    "sctr": process_metrics['sctr'][t1].get(file, 0),
                    "ncomm": process_metrics['ncomm'][t1].get(file, 0),
                    "nadev": process_metrics['nadev'][t1].get(file, 0),
                    "nddev": process_metrics['nddev'][t1].get(file, 0),
                    "nsctr": process_metrics['nsctr'][t1].get(file, 0),
                    "oexp": process_metrics['oexp'][t1].get(file, 0),
                    "exp": process_metrics['exp'][t1].get(file, 0),
                    'Change_entropy': HCPF2.get(file, 0),
                    'Co_change_entropy': file_co_change_entropy.get(file, 0),
                    'Bugs': file_bugs.get(file, 0),
                    'Minor': file_bugs_priority.get(file, {}).get('Minor', 0),
                    'Major': file_bugs_priority.get(file, {}).get('Major', 0),
                    'Critical': file_bugs_priority.get(file, {}).get('Critical', 0),
                    'Trivial': file_bugs_priority.get(file, {}).get('Trivial', 0),
                    'Blocker': file_bugs_priority.get(file, {}).get('Blocker', 0)
                }
                release_data.append(row)

            # Convert to DataFrame and store
            release_df = pd.DataFrame(release_data)
            all_data.append(release_df)

        # Concatenate data for training and testing
        if len(all_data) > 1:
            train_data = pd.concat(all_data[:-1], axis=0)  # All except last for training
            test_data = all_data[-1]  # Last release for testing
            train_test = pd.concat(all_data, axis=0)
        else:
            print("Not enough releases for train-test split.")
            return

        # Save the CSVs in the required order
        required_columns = ['File', 'comm', 'adev', 'ddev', 'add', 'del', 'own', 'minor', 'sctr', 'ncomm', 'nadev',
                            'nddev', 'nsctr', 'oexp', 'exp', 'Change_entropy', 'Co_change_entropy',
                            'Bugs', 'Minor', 'Major', 'Critical', 'Trivial', 'Blocker'
                            ]
        train_data = train_data[required_columns]
        test_data = test_data[required_columns]
        train_test_data = train_test[required_columns]

        train_data.to_csv(f'{project}_training_file_entropy_bugs_bug_priorities_non_fatty_h1_final.csv', index=False)
        test_data.to_csv(f'{project}_test_file_entropy_bugs_bug_priorities_non_fatty_h1_final.csv', index=False)
        train_test_data.to_csv(f'{project}_train_test_file_entropy_bugs_bug_priorities_non_fatty_h1_final.csv',
                               index=False)

        print("Training and test CSVs generated successfully.")


if __name__ == "__main__":
    st = datetime.now()
    print(f"start time : {datetime.now()}")
    t = Task2()
    print(f"\ncommits files building")
    t.build_commit_set_of_files()
    print(f"\ncommit time and project name building")
    t.build_commit_time()
    print(f"\nfile path building")
    t.build_file_path()
    project_release = {
        "derby": ["10.3.1.4", "10.5.1.1"],
        "activemq": ["activemq-5.0.0", "activemq-5.1.0", "activemq-5.2.0", "activemq-5.3.0", "activemq-5.5.0"],
        "pdfbox": ["1.5.0", "1.7.0", "1.8.0", "2.0.0"],
        "pig": ["release-0.6.0", "release-0.7.0", "release-0.8.0", "release-0.9.0"],
        "kafka": ["0.10.0.0", "0.11.0.0"],
        "maven": ["maven-3.1.0", "maven-3.3.9", "maven-3.5.0"],
        "struts": ["STRUTS_2_3_28", "STRUTS_2_3_32"],
        "nifi": ["nifi-0.5.0", "nifi-0.6.0", "nifi-0.7.0"]
    }
    for p in project_release:
        print(f"project = {p}")
        print(f"\ncommit src files building")
        t.build_commit_set_of_src_files(p)
        print(f"\ntime and graph building")
        releases = project_release[p]
        t.build_time(p, releases)
        print(f"--------------------------------------------------")
    print(f"Start time : {st}")
    print(f"End time : {datetime.now()}")
