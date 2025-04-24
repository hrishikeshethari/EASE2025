
import pickle

from pymongo import MongoClient
import networkx as nx
import datetime
from datetime import datetime, timedelta

from Build_reverse_identity_dictionary import Build_reverse_identity_dictionary


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
        self.file_path_records = list(self.file_path.find({}, {"_id": 1, "path": 1}))
        self.commit_issue_records = list(self.commit_project.find({}, {"_id": 1, "linked_issue_ids": 1}))
        self.issue_data = self.db["issue"]
        self.issue_records = list(self.issue_data.find({}, {}))
        self.commit_issue_data = dict()
        self.issue_issue_type = dict()
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
        self.refactoring_types = list()
        self.file_bugs = dict()
        self.D = dict()

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

        for element in self.issue_records:
            # print(element.keys())
            for key in element.keys():
                if key == 'priority':
                    issue = element["_id"]
                    issue_types = element["issue_type"]
                    status = element['status']
                    if issue_types == 'Bug':
                        if status == 'Resolved' or status == 'Closed':
                            if 'affects_versions' in element.keys():
                                if issue not in self.issue_issue_type.keys():
                                    self.issue_issue_type[issue] = list()
                                if issue_types not in self.issue_issue_type[issue]:
                                    self.issue_issue_type[issue].append(issue_types)

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
                    ordered_pairs = {(x, y) for x in files for y in files if x != y}
                    self.graph.add_edges_from(ordered_pairs)

        with open(f'{project}_{t1}_file_bugs_ids_release_entropy_non_fatty.pkl', 'rb') as f:
            file_bugs = pickle.load(f)

        total_issues = list()
        for commit in self.commit_issue_data:
            if commit in self.graph_data:
                issues = self.commit_issue_data[commit]
                for issue in issues:
                    if issue in self.issue_issue_type:
                        if issue not in total_issues:
                            total_issues.append(issue)

        total_files_changed = list()
        for commit in self.graph_data:
            if commit in self.commit_set_of_src_files:
                files = self.commit_set_of_src_files[commit]
                for f in files:
                    if f not in total_files_changed:
                        total_files_changed.append(f)

        commit_times = [self.commit_time[commit] for commit in self.graph_data if commit in self.commit_time]

        # Find the first and last commit times
        if commit_times:
            first_commit_time = min(commit_times)
            last_commit_time = max(commit_times)
        else:
            first_commit_time = None
            last_commit_time = None

        print(f"First Commit Time: {first_commit_time}")
        print(f"Last Commit Time: {last_commit_time}")

        file_issues = list()
        for file in file_bugs:
            bugs = file_bugs[file]
            for bug in bugs:
                if bug in total_issues:
                    if bug not in file_issues:
                        file_issues.append(bug)

        defective_rate = len(file_issues)/len(total_files_changed)
        print(f"commits in {project} | {t1} = {len(self.graph_data)}")
        print(f"defective files of {project} | {t1} = {len(file_issues)}")
        print(f"total defects of {project} | {t1} = {len(total_issues)}")
        print(f"defective rate of {project} | {t1} = {defective_rate}")
        result_file2.write(str(project))
        result_file2.write(",")
        result_file2.write(str(t1))
        result_file2.write(",")
        result_file2.write(str(len(self.graph.edges())))
        result_file2.write(",")
        result_file2.write(str(len(self.graph.nodes())))
        result_file2.write(",")
        result_file2.write(str(len(total_issues)))
        result_file2.write(",")
        result_file2.write(str(defective_rate))
        result_file2.write(",")
        result_file2.write(str(first_commit_time))
        result_file2.write(",")
        result_file2.write(str(last_commit_time))
        result_file2.write("\n")


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
        "nifi": ["nifi-0.5.0", "nifi-0.6.0", "nifi-0.7.0"]}

    csv_file = "entropy_project_table.csv"
    result_file2 = open(csv_file, "w")
    result_file2.write("Project")
    result_file2.write(",")
    result_file2.write("Release")
    result_file2.write(",")
    result_file2.write("Edges")
    result_file2.write(",")
    result_file2.write("Nodes")
    result_file2.write(",")
    result_file2.write("Defects")
    result_file2.write(",")
    result_file2.write("Defect Ratio")
    result_file2.write(",")
    result_file2.write("Time of First Commit")
    result_file2.write(",")
    result_file2.write("Time of Last Commit")
    result_file2.write("\n")
    for p in project_release:
        print(f"project = {p}")
        print(f"\ncommit src files building")
        t.build_commit_set_of_src_files(p)
        for rw in project_release[p]:
            print(f"\ntime and graph building")
            t.print_data(p, rw)
    print(f"Start time : {st}")
    print(f"End time : {datetime.now()}")
