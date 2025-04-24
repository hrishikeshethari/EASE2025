import collections
import itertools
import math
import os
import pickle
from collections import Counter
from collections import defaultdict

import treelib
from dateutil.relativedelta import relativedelta
from matplotlib import pyplot as plt
from pymongo import MongoClient
import networkx as nx
import datetime
from datetime import datetime, timedelta, date
from itertools import combinations, chain
from Build_reverse_identity_dictionary import Build_reverse_identity_dictionary
import re


class Task2:

    def __init__(self):
        self.client = MongoClient("mongodb://localhost:27017/")
        self.db = self.client["smartshark"]
        self.commit_project = self.db["commit_with_project_info"]
        self.issue_data = self.db["issue"]
        self.commit_project_records = list(self.commit_project.find({}, {}))
        self.commit_file = self.db["file_action"]
        self.file_path = self.db["file"]
        self.commit_file_records = list(self.commit_file.find({}, {"file_id": 1, "commit_id": 1}))
        self.file_path_records = list(self.file_path.find({}, {"_id": 1, "path": 1}))
        self.commit_issue_records = list(self.commit_project.find({}, {"_id": 1, "linked_issue_ids": 1}))
        self.issue_records = list(self.issue_data.find({}, {}))
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
        self.path_tree = nx.DiGraph()
        self.node_mapping = dict()
        self.file_level1 = dict()
        self.g_test = nx.Graph()
        self.file_extension = dict()
        self.final_file_paths = set()
        self.file_path_data_unique = dict()
        self.graph_filename = nx.Graph()
        self.tree = treelib.Tree()
        self.file_filename = dict()
        self.filename_file = dict()
        self.level_entropy = dict()
        self.file_entropy_difference = dict()
        self.src_file_commits = dict()
        self.commit_issue_data = dict()
        self.src_file_issue = dict()
        self.issue_issue_type = dict()
        self.file_bugs = dict()
        self.issue_issue_priority = dict()
        self.file_bugs_priority = dict()
        self.issue_bugs = set()
        self.issue_commits = dict()
        self.issues = set()
        self.project_release_bugs = dict()
        self.vector_centrality = dict()
        self.vector_centrality_betweeness = dict()
        self.vector_centrality_closeness = dict()
        self.vector_centrality_degree = dict()
        self.D = dict()
        self.graph = nx.Graph()
        self.graph_data = dict()
        self.graph_data_hyper = dict()
        self.hyper_graph = dict()
        self.sorted_centrality = dict()
        self.sorted_betweeness = dict()
        self.sorted_closeness = dict()
        self.sorted_degree = dict()

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

    def build_file_commits(self):
        self.src_file_commits.clear()
        for commit in self.commit_set_of_src_files:
            # print(commit)
            if commit in self.graph_data:
                files = self.commit_set_of_src_files[commit]
                r = len(files)
                for i in range(r):
                    if files[i] not in self.src_file_commits.keys():
                        self.src_file_commits[files[i]] = list()
                    if commit not in self.src_file_commits[files[i]]:
                        self.src_file_commits[files[i]].append(commit)
        print(f"src file commits = {len(self.src_file_commits)}")

    def build_commit_issue(self):
        for element in self.commit_issue_records:
            try:
                issue_id = element["linked_issue_ids"]
            except KeyError:
                issue_id = "0"
            if issue_id and issue_id != '0':
                commit_id = element["_id"]
                if commit_id not in self.commit_issue_data.keys():
                    self.commit_issue_data[commit_id] = list()
                if issue_id not in self.commit_issue_data[commit_id]:
                    self.commit_issue_data[commit_id] = issue_id

    def build_file_issue(self):
        self.src_file_issue.clear()
        for file in self.src_file_commits.keys():
            commit = self.src_file_commits[file]
            r = len(commit)
            for i in range(r):
                # print(f"commit[i] = {commit[i]}")
                if commit[i] in self.commit_issue_data.keys():
                    issue = self.commit_issue_data[commit[i]]
                    if file not in self.src_file_issue.keys():
                        self.src_file_issue[file] = list()
                    if issue not in self.src_file_issue[file]:
                        self.src_file_issue[file].append(issue)

        for file in self.src_file_issue.keys():
            self.src_file_issue[file] = list(itertools.chain.from_iterable(self.src_file_issue[file]))

    def build_issue_issue_type(self):
        for element in self.issue_records:
            # print(element.keys())
            for key in element.keys():
                if key == 'priority':
                    issue = element["_id"]
                    issue_types = element["issue_type"]
                    issue_priority = element["priority"]
                    status = element['status']
                    if issue_types == 'Bug':
                        if status == 'Resolved' or status == 'Closed':
                            if 'affects_versions' in element.keys():
                                if issue not in self.issue_issue_type.keys():
                                    self.issue_issue_type[issue] = list()
                                    self.issue_issue_priority[issue] = list()
                                if issue_types not in self.issue_issue_type[issue]:
                                    self.issue_issue_type[issue].append(issue_types)
                                if issue_priority not in self.issue_issue_priority[issue]:
                                    self.issue_issue_priority[issue].append(issue_priority)

    def build_project_issues_bugs(self, project):
        ci = 0
        cb = 0
        for commit in self.commit_issue_data:
            if self.commit_project_data[commit] == project:
                ci = ci + 1
                if self.commit_issue_data[commit][0] in self.issue_issue_type.keys():
                    if self.issue_issue_type[self.commit_issue_data[commit][0]][0] == 'Bug':
                        cb = cb + 1
        print(f"project = {project} | issues = {ci} | bugs = {cb}")

    def build_file_number_of_bugs(self, project, t1):
        self.file_bugs.clear()
        file_issue_type = dict()
        file_issue_bugs = dict()

        for file in self.src_file_issue:
            issue = self.src_file_issue[file]
            r = len(issue)
            c = 0
            if file not in file_issue_bugs.keys():
                file_issue_bugs[file] = list()
            for i in range(r):
                if issue[i] in self.issue_issue_type.keys():
                    issue_type = self.issue_issue_type[issue[i]]
                    if issue_type[0] == 'Bug':
                        c = c + 1
                        if issue[i] not in file_issue_bugs[file]:
                            file_issue_bugs[file].append(issue[i])
            if file not in self.file_bugs.keys():
                if c > 0:
                    self.file_bugs[file] = c

        for file in self.src_file_issue:
            issue = self.src_file_issue[file]
            r = len(issue)
            c = 0
            if file not in file_issue_type.keys():
                file_issue_type[file] = dict()
            for i in range(r):
                if issue[i] in self.issue_issue_type.keys():
                    issue_type = self.issue_issue_type[issue[i]]
                    issue_priority = self.issue_issue_priority[issue[i]]
                    if issue_type[0] not in file_issue_type[file].keys():
                        file_issue_type[file][issue_type[0]] = 1
                    else:
                        file_issue_type[file][issue_type[0]] = file_issue_type[file][issue_type[0]] + 1
                    if issue_type[0] == 'Bug':
                        c = c + 1
                        if file not in self.file_bugs_priority.keys():
                            self.file_bugs_priority[file] = dict()
                        if issue_priority[0] not in self.file_bugs_priority[file].keys():
                            self.file_bugs_priority[file][issue_priority[0]] = 1
                        else:
                            self.file_bugs_priority[file][issue_priority[0]] = self.file_bugs_priority[file][
                                                                                   issue_priority[0]] + 1

        for f in self.file_bugs_priority:
            print(f"files = {f} | path = {self.file_extension[f]} | priority = {self.file_bugs_priority[f]}")
        with open(f'{project}_{t1}_file_bugs_release_entropy_non_fatty.pkl', 'wb') as f:
            pickle.dump(self.file_bugs, f)

        with open(f'{project}_{t1}_file_bugs_ids_release_entropy_non_fatty.pkl', 'wb') as f:
            pickle.dump(file_issue_bugs, f)

        with open(f'{project}_{t1}_file_bugs_priority_release_entropy_non_fatty.pkl', 'wb') as f:
            pickle.dump(self.file_bugs_priority, f)

        with open(f'{project}_{t1}_file_issue_types_release_entropy_non_fatty.pkl', 'wb') as f:
            pickle.dump(file_issue_type, f)


if __name__ == "__main__":
    st = datetime.now()
    print(f"start time : {datetime.now()}")
    t = Task2()

    project_release = {
        "derby": ["10.3.1.4", "10.5.1.1"],
        "activemq": ["activemq-5.0.0", "activemq-5.1.0", "activemq-5.2.0", "activemq-5.3.0", "activemq-5.5.0"],
        "pdfbox": ["1.5.0", "1.7.0", "1.8.0", "2.0.0"],
        "pig": ["release-0.6.0", "release-0.7.0", "release-0.8.0", "release-0.9.0"],
        "kafka": ["0.10.0.0", "0.11.0.0"],
        "maven": ["maven-3.1.0", "maven-3.3.9", "maven-3.5.0"],
        "struts": ["STRUTS_2_3_28", "STRUTS_2_3_32"],
        "nifi": ["nifi-0.5.0", "nifi-0.6.0", "nifi-0.7.0"]}

    print(f"\ncommits files building")
    t.build_commit_set_of_files()
    print(f"\ncommit time and project name building")
    t.build_commit_time()
    print(f"\nfile path building")
    t.build_file_path()
    t.build_commit_issue()
    t.build_issue_issue_type()
    for p in project_release:
        print(f"project = {p}")
        print(f"\ncommit src files building")
        t.build_commit_set_of_src_files(p)
        print(f"\ntime and graph building")
        releases = project_release[p]
        for i1 in releases:
            print(f"{p} - {i1}")
            t.build_time(p, i1)
            t.build_file_commits()
            t.build_file_issue()
            t.build_file_number_of_bugs(p, i1)
            print(f"---------------------------------------------------------")
    print(f"Start time : {st}")
    print(f"End time : {datetime.now()}")
