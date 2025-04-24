import collections
import csv
import gc
import itertools
import math
import os
import pickle
import statistics
from collections import Counter
from collections import defaultdict
import hypernetx as hnx
import numpy
import numpy as np
import openpyxl
from dateutil.relativedelta import relativedelta
from matplotlib import pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.ticker import MultipleLocator
from pymongo import MongoClient
import networkx as nx
import datetime
from datetime import datetime, timedelta
from itertools import combinations, chain
from scipy.sparse import csr_matrix
import networkit as nk
from scipy.stats import pearsonr
from memory_profiler import profile, memory_usage

from Build_reverse_identity_dictionary import Build_reverse_identity_dictionary
import re


class Task:

    def __init__(self):
        self.node_mapping = dict()
        self.sparse_matrix = None
        self.client = MongoClient("mongodb://localhost:27017/")
        self.db = self.client["smartshark"]
        self.commit_project = self.db["commit_with_project_info"]
        self.commit_project_records = list(self.commit_project.find({}, {}))
        self.commit_file = self.db["file_action"]
        self.file_path = self.db["file"]
        self.commit_file_records = list(self.commit_file.find({}, {"file_id": 1, "commit_id": 1}))
        self.file_lines_records = list(self.commit_file.find({}, {"file_id": 1, "lines_added": 1, "lines_deleted": 1}))
        self.commit_lines_records = list(
            self.commit_file.find({}, {"commit_id": 1, "lines_added": 1, "lines_deleted": 1}))
        self.file_path_records = list(self.file_path.find({}, {"_id": 1, "path": 1}))
        self.refactoring_data = self.db["refactoring"]
        self.refactoring_records = list(
            self.refactoring_data.find({}, {"_id": 1, "commit_id": 1, "type": 1, "description": 1}))
        self.BRID = Build_reverse_identity_dictionary()
        self.BRID.reading_identity_and_people_and_building_reverse_identity_dictionary()
        self.commit_set_of_files = dict()
        self.commit_set_of_src_files = dict()
        self.commit_project_data = dict()
        self.commit_time_unsort = dict()
        self.commit_time = dict()
        self.file_path_data = dict()
        self.src_files_set = set()
        self.file_extension = dict()
        self.graph = nx.Graph()
        self.graph_data = dict()
        self.graph_data_hyper = dict()
        self.hyper_graph = dict()
        self.sorted_centrality = dict()
        self.sorted_betweeness = dict()
        self.sorted_closeness = dict()
        self.sorted_degree = dict()
        self.commit_file_distribution = dict()
        self.refactoring_types = list()
        self.vector_centrality = dict()
        self.vector_centrality_betweeness = dict()
        self.vector_centrality_closeness = dict()
        self.vector_centrality_degree = dict()
        self.file_bugs = dict()
        self.D = dict()
        self.src_file_commits = dict()
        self.commit_author_data = dict()
        self.metrics = {
            "Size": {},
            "Ties": {},
            "Pairs": {},
            "Density": {},
            "WeakComp": {},
            "nWeakComp": {},
            "TwoStepReach": {},
            "ReachEfficency": {},
            "Brokerage": {},
            "nBrokerage": {},
            "EgoBetween": {},
            "nEgoBetween": {},
            "EffSize": {},
            "Efficiency": {},
        }
        self.process_metrics = {
            "comm": {},
            "adev": {},
            "ddev": {},
            "add": {},
            "del": {},
            "own": {},
            "minor": {},
            "sctr": {},
            "ncomm": {},
            "nadev": {},
            "nddev": {},
            "nsctr": {},
            "oexp": {},
            "exp": {}
        }

    def build_commit_set_of_files(self):
        for element in self.commit_file_records:
            file_id = element["file_id"]
            commit_id = element["commit_id"]
            if commit_id not in self.commit_set_of_files.keys():
                self.commit_set_of_files[commit_id] = set()
            self.commit_set_of_files[commit_id].add(file_id)

        for element in self.commit_project_records:
            commit_id = element["_id"]
            author_id = self.BRID.reverse_identity_dict[element["author_id"]]
            if commit_id not in self.commit_author_data.keys():
                self.commit_author_data[commit_id] = list()
            if commit_id not in self.commit_author_data[commit_id]:
                self.commit_author_data[commit_id].append(author_id)

    def build_commit_set_of_src_files(self, project):
        self.commit_set_of_src_files.clear()
        for commit in self.commit_set_of_files:
            if self.commit_project_data[commit] == project:
                files = self.commit_set_of_files[commit]
                files = list(files)
                r = len(files)
                for i in range(r):
                    if files[i] in self.file_path_data:
                        path1 = self.file_path_data[files[i]]
                        pos1 = path1.rfind('/')
                        if pos1 != -1:
                            trim1 = path1[0:pos1:1]
                            modules1 = trim1.split("/")
                            # if "src" in modules1 and files[i] in self.src_files_set:
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

    def print_data(self, project, t1):
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

    def create_normal_graph(self, t1, project):
        self.graph.clear()
        self.graph_data.clear()
        with open(f'project_release_commits_final_new.pkl', 'rb') as f:
            project_release_commits = pickle.load(f)
        for pro in project_release_commits:
            if pro == project:
                for rel in project_release_commits[pro].keys():
                    if rel == t1:
                        for commit in project_release_commits[pro][rel]:
                            self.graph_data[commit] = self.commit_time[commit]

        for commit in self.commit_set_of_src_files:
            if self.commit_project_data[commit] == project:
                if commit in self.graph_data:
                    files = self.commit_set_of_src_files[commit]
                    if len(files) <= 30:
                        ordered_pairs = {(x, y) for x in files for y in files if x != y}
                        self.graph.add_edges_from(ordered_pairs)

        with open(f'{project}_{t1}_graph_non_fatty.pkl', 'wb') as f1:
            pickle.dump(self.graph, f1)

        src_file_commits = dict()
        for commit in self.commit_set_of_src_files:
            if commit in self.graph_data:
                files = self.commit_set_of_src_files[commit]
                r = len(files)
                for i in range(r):
                    if files[i] not in src_file_commits.keys():
                        src_file_commits[files[i]] = list()
                    if commit not in src_file_commits[files[i]]:
                        src_file_commits[files[i]].append(commit)

        src_file_authors = dict()
        for file in src_file_commits.keys():
            commit = src_file_commits[file]
            r = len(commit)
            for i in range(r):
                if commit[i] in self.commit_author_data.keys():
                    author = self.commit_author_data[commit[i]]
                    if file not in src_file_authors.keys():
                        src_file_authors[file] = list()
                    if author not in src_file_authors[file]:
                        src_file_authors[file].append(author)
        for file in src_file_authors.keys():
            src_file_authors[file] = list(itertools.chain.from_iterable(src_file_authors[file]))

        time_list_graph = list()
        time_list = list()
        for commit in self.commit_time:
            if self.commit_project_data[commit] == project:
                time_list.append(self.commit_time[commit])
        t1min = min(time_list)
        t7max = max(time_list)
        for commit in self.graph_data:
            time_list_graph.append(self.graph_data[commit])
        tgmax = max(time_list_graph)

        src_file_commits_distinct = dict()
        for commit in self.commit_set_of_src_files:
            if t1min <= self.commit_time[commit] <= tgmax:
                files = self.commit_set_of_src_files[commit]
                r = len(files)
                for i in range(r):
                    if files[i] not in src_file_commits_distinct.keys():
                        src_file_commits_distinct[files[i]] = list()
                    if commit not in src_file_commits_distinct[files[i]]:
                        src_file_commits_distinct[files[i]].append(commit)
        src_file_authors_distinct = dict()
        for file in src_file_commits_distinct.keys():
            commit = src_file_commits_distinct[file]
            r = len(commit)
            for i in range(r):
                if commit[i] in self.commit_author_data.keys():
                    author = self.commit_author_data[commit[i]]
                    if file not in src_file_authors_distinct.keys():
                        src_file_authors_distinct[file] = list()
                    if author not in src_file_authors_distinct[file]:
                        src_file_authors_distinct[file].append(author)
        for file in src_file_authors_distinct.keys():
            src_file_authors_distinct[file] = list(itertools.chain.from_iterable(src_file_authors_distinct[file]))

        commit_lines = dict()
        commit_lines_enitre = dict()
        for element in self.commit_lines_records:
            commit = element["commit_id"]
            lines_added = element["lines_added"]
            lines_deleted = element["lines_deleted"]
            if commit in self.graph_data:
                if commit not in commit_lines.keys():
                    commit_lines[commit] = 0
                commit_lines[commit] = commit_lines[commit] + lines_added + lines_deleted
            if t1min <= self.commit_time[commit] <= t7max:
                if commit not in commit_lines_enitre.keys():
                    commit_lines_enitre[commit] = 0
                commit_lines_enitre[commit] = commit_lines_enitre[commit] + lines_added + lines_deleted

        author_lines = dict()
        for commit in commit_lines:
            authors = self.commit_author_data[commit]
            for a in authors:
                if a not in author_lines.keys():
                    author_lines[a] = 0
                author_lines[a] = author_lines[a] + commit_lines[commit]

        author_lines_enitre = dict()
        for commit in commit_lines_enitre:
            authors = self.commit_author_data[commit]
            for a in authors:
                if a not in author_lines_enitre.keys():
                    author_lines_enitre[a] = 0
                author_lines_enitre[a] = author_lines_enitre[a] + commit_lines_enitre[commit]

        src_file_lines_added = dict()
        src_file_lines_deleted = dict()
        total_lines_added = 0
        total_lines_deleted = 0
        for element in self.file_lines_records:
            f = element["file_id"]
            lines_added = element["lines_added"]
            lines_deleted = element["lines_deleted"]
            total_lines_added = total_lines_added + lines_added
            total_lines_deleted = total_lines_deleted + lines_deleted
            if f in src_file_commits:
                if f not in src_file_lines_added.keys():
                    src_file_lines_added[f] = 0
                src_file_lines_added[f] = src_file_lines_added[f] + lines_added
                if f not in src_file_lines_deleted.keys():
                    src_file_lines_deleted[f] = 0
                src_file_lines_deleted[f] = src_file_lines_deleted[f] + lines_deleted

        HCPF2 = dict()
        file_commit_counts = {file: len(commits) for file, commits in src_file_commits.items()}
        total_changes = sum(file_commit_counts.values())
        probabilities = [count / total_changes for count in file_commit_counts.values()]
        entropy_Hi = -sum(p * math.log2(p) for p in probabilities if p > 0)
        HCPF2 = {file: (count / total_changes) * entropy_Hi for file, count in file_commit_counts.items()}

        comm = dict()
        adev = dict()
        ddev = dict()
        sctr = dict()
        ncomm = dict()
        nadev = dict()
        nddev = dict()
        nsctr = dict()
        add = dict()
        dele = dict()
        own = dict()
        minor = dict()
        oexp = dict()
        exp = dict()
        for i, f in enumerate(self.graph.nodes()):
            print(f"release = {t1} | {i + 1}/{len(self.graph.nodes())} - {datetime.now()}")
            comm[f] = len(src_file_commits[f])
            print(f"comm of {f} = {comm[f]}")
            adev[f] = len(src_file_authors[f])
            print(f"adev of {f} = {adev[f]}")
            ddev[f] = len(src_file_authors_distinct[f])
            print(f"ddev of {f} = {ddev[f]}")
            sctr[f] = HCPF2[f]
            print(f"sctr of {f} = {sctr[f]}")

            neighbors = list(self.graph.neighbors(f))
            commn = 0
            adevn = 0
            ddevn = 0
            sctrn = 0
            if len(neighbors) > 0:
                for n in neighbors:
                    if n in src_file_commits.keys():
                        commn = commn + len(src_file_commits[n])
                        adevn = adevn + len(src_file_authors[n])
                        ddevn = ddevn + len(src_file_authors_distinct[n])
                        sctrn = sctrn + HCPF2[n]
                    else:
                        print(f"non src file = {n}")
                ncomm[f] = commn / len(neighbors)
                nadev[f] = adevn / len(neighbors)
                nddev[f] = ddevn / len(neighbors)
                nsctr[f] = sctrn / len(neighbors)
            else:
                ncomm[f] = 0
                nadev[f] = 0
                nddev[f] = 0
                nsctr[f] = 0
            print(f"ncomm of {f} = {ncomm[f]}")
            print(f"nadev of {f} = {nadev[f]}")
            print(f"nddev of {f} = {nddev[f]}")
            print(f"nsctr of {f} = {nsctr[f]}")

            add[f] = src_file_lines_added[f] / total_lines_added
            print(f"add of {f} = {add[f]}")
            dele[f] = src_file_lines_deleted[f] / total_lines_deleted
            print(f"del of {f} = {dele[f]}")

            authors = src_file_authors[f]
            max_lines = 0
            total_lines = 0
            lines_file = src_file_lines_added[f] + src_file_lines_deleted[f]
            authors_list = list()
            exp_author_lines = 0
            mean_list = list()
            for a in authors:
                if max_lines < author_lines[a]:
                    max_lines = author_lines[a]
                    exp_author_lines = author_lines_enitre[a]
                total_lines = total_lines + author_lines[a]

                mean_list.append(author_lines[a])

                if author_lines[a] <= (lines_file * 5) / 100:
                    authors_list.append(a)

            own[f] = max_lines / total_lines
            print(f"own of {f} = {own[f]}")
            minor[f] = len(authors_list)
            print(f"minor of {f} = {minor[f]}")

            oexp[f] = max_lines / exp_author_lines
            print(f"oexp of {f} = {oexp[f]}")
            exp[f] = statistics.geometric_mean(mean_list)
            print(f"exp of {f} = {exp[f]}")

        self.process_metrics['comm'][t1] = comm
        self.process_metrics['adev'][t1] = adev
        self.process_metrics['ddev'][t1] = ddev
        self.process_metrics['sctr'][t1] = sctr
        self.process_metrics['ncomm'][t1] = ncomm
        self.process_metrics['nadev'][t1] = nadev
        self.process_metrics['nddev'][t1] = nddev
        self.process_metrics['nsctr'][t1] = nsctr
        self.process_metrics['add'][t1] = add
        self.process_metrics['del'][t1] = dele
        self.process_metrics['own'][t1] = own
        self.process_metrics['minor'][t1] = minor
        self.process_metrics['oexp'][t1] = oexp
        self.process_metrics['exp'][t1] = exp


if __name__ == "__main__":
    st = datetime.now()
    print(f"start time : {datetime.now()}")
    t = Task()
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
        for rw in project_release[p]:
            print(f"\ntime and graph building")
            t.print_data(p, rw)
        with open(f'{p}_process_metrics_release_new_entropy_non_fatty.pkl', 'wb') as f:
            pickle.dump(t.process_metrics, f)
        t.process_metrics['comm'].clear()
        t.process_metrics['adev'].clear()
        t.process_metrics['ddev'].clear()
        t.process_metrics['sctr'].clear()
        t.process_metrics['ncomm'].clear()
        t.process_metrics['nadev'].clear()
        t.process_metrics['nddev'].clear()
        t.process_metrics['nsctr'].clear()
        t.process_metrics['add'].clear()
        t.process_metrics['del'].clear()
        t.process_metrics['own'].clear()
        t.process_metrics['minor'].clear()
        t.process_metrics['oexp'].clear()
        t.process_metrics['exp'].clear()
    print(f"Start time : {st}")
    print(f"End time : {datetime.now()}")
