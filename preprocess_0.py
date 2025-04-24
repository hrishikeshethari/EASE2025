import pickle
from collections import defaultdict

from pymongo import MongoClient
from datetime import datetime
from Build_reverse_identity_dictionary import Build_reverse_identity_dictionary


class Task2:

    def __init__(self):
        self.client = MongoClient("mongodb://localhost:27017/")
        self.db = self.client["smartshark"]
        self.commit_project = self.db["commit_with_project_info"]
        self.release_tags = self.db["tag_with_project_info"]
        self.commit_project_records = list(self.commit_project.find({}, {}))
        self.release_commits_records = list(self.release_tags.find({}, {}))
        self.BRID = Build_reverse_identity_dictionary()
        self.BRID.reading_identity_and_people_and_building_reverse_identity_dictionary()
        self.project_commit_data = defaultdict()
        self.commit_project_data = {}
        self.commit_release = defaultdict()
        self.child_to_parents = defaultdict()
        self.revision_to_commits = defaultdict()
        self.release_commits = defaultdict()
        self.project_release_commits = defaultdict()
        self.commit_time_tag = defaultdict()
        self.commit_time_tag_unsort = defaultdict()
        self.commit_time = defaultdict()
        self.commit_time_unsort = defaultdict()
        self.release_time = defaultdict()
        self.project_release_times = defaultdict()

    def build_commit_time(self):
        for element in self.commit_project_records:
            commit_id = element["_id"]
            project_name = element["project_name_info"]["name"]
            c_time = element["committer_date"]
            if project_name not in self.project_commit_data:
                self.project_commit_data[project_name] = list()
            if commit_id not in self.project_commit_data[project_name]:
                self.project_commit_data[project_name].append(commit_id)
            if commit_id not in self.commit_project_data.keys():
                self.commit_project_data[commit_id] = project_name
            if commit_id not in self.commit_time_unsort.keys():
                self.commit_time_unsort[commit_id] = c_time
        self.commit_time = {key: value for key, value in sorted(self.commit_time_unsort.items(),
                                                                key=lambda item: item[1])}

    def build_commit_time_tag(self):
        for element in self.release_commits_records:
            if "date" in element.keys():
                release = element["name"]
                commit_id = element["commit_id"]
                c_time = element["date"]
                if commit_id not in self.commit_time_tag_unsort.keys():
                    self.commit_time_tag_unsort[commit_id] = c_time
                if release not in self.release_commits:
                    self.release_commits[release] = list()
                if commit_id not in self.release_commits[release]:
                    self.release_commits[release].append(commit_id)
                if release not in self.release_time:
                    self.release_time[release] = list()
                if c_time not in self.release_time[release]:
                    self.release_time[release].append(c_time)
                if commit_id not in self.commit_release:
                    self.commit_release[commit_id] = list()
                if release not in self.commit_release[commit_id]:
                    self.commit_release[commit_id].append(release)
        self.commit_time_tag = {key: value for key, value in sorted(self.commit_time_tag_unsort.items(),
                                                                key=lambda item: item[1])}

    def build_project_release_times(self):
        for element in self.release_commits_records:
            if "date" in element.keys():
                release = element["name"]
                project = element["project_name"]
                c_time = element["date"]
                if project not in self.project_release_times:
                    self.project_release_times[project] = defaultdict()
                if release not in self.project_release_times[project]:
                    self.project_release_times[project][release] = list()
                if c_time not in self.project_release_times[project][release]:
                    self.project_release_times[project][release].append(c_time)

    def build_project_release_time_commit(self):
        projects = {
            "derby": ["10.1.3.1", "10.2.1.6", "10.3.1.4", "10.5.1.1"],
            "activemq": ["activemq-4.1.1", "activemq-5.0.0", "activemq-5.1.0", "activemq-5.2.0", "activemq-5.3.0",
                         "activemq-5.5.0"],
            "pdfbox": ["1.1.0", "1.5.0", "1.7.0", "1.8.0", "2.0.0"],
            "pig": ["release-0.5.0", "release-0.6.0", "release-0.7.0", "release-0.8.0", "release-0.9.0"],
            "kafka": ["0.9.0.0", "0.10.0.0", "0.10.1.0", "0.10.2.0", "0.11.0.0"],
            "maven": ["maven-3.0", "maven-3.1.0", "maven-3.3.9", "maven-3.5.0"],
            "struts": ["STRUTS_2_3_20", "STRUTS_2_3_24", "STRUTS_2_3_28", "STRUTS_2_3_32"],
            "nifi": ["nifi-0.4.0", "nifi-0.5.0", "nifi-0.6.0", "nifi-0.7.0"]
        }
        for project in projects:
            commits = self.project_commit_data[project]
            releases = projects[project]
            for i in range(len(releases) - 1):
                r1 = releases[i]
                r2 = releases[i + 1]
                c1 = self.release_commits[r1]
                c2 = self.release_commits[r2]
                print(f"r1 = {r1} | c1 = {c1}")
                print(f"r2 = {r2} | c2 = {c2}")
                if project == "pdfbox":
                    if r1 == '1.1.0':
                        t1 = datetime(2010, 3, 29, 00, 00, 00)
                        t2 = datetime(2011, 3, 3, 00, 00, 00)
                    elif r1 == '1.5.0':
                        t1 = datetime(2011, 3, 3, 00, 00, 00)
                        t2 = datetime(2012, 5, 28, 00, 00, 00)
                    elif r1 == '1.7.0':
                        t1 = datetime(2012, 5, 28, 00, 00, 00)
                        t2 = datetime(2013, 3, 22, 00, 00, 00)
                    elif r1 == '1.8.0':
                        t1 = datetime(2013, 3, 22, 00, 00, 00)
                        t2 = datetime(2016, 3, 18, 00, 00, 00)
                elif project == "pig":
                    if r1 == 'release-0.5.0':
                        t1 = datetime(2009, 10, 29, 00, 00, 00)
                        t2 = datetime(2010, 3, 1, 00, 00, 00)
                    elif r1 == 'release-0.6.0':
                        t1 = datetime(2010, 3, 1, 00, 00, 00)
                        t2 = datetime(2010, 5, 13, 00, 00, 00)
                    elif r1 == 'release-0.7.0':
                        t1 = datetime(2010, 5, 13, 00, 00, 00)
                        t2 = datetime(2010, 12, 17, 00, 00, 00)
                    elif r1 == 'release-0.8.0':
                        t1 = datetime(2010, 12, 17, 00, 00, 00)
                        t2 = datetime(2011, 7, 29, 00, 00, 00)
                elif project == "derby":
                    if r1 == '10.1.3.1':
                        t1 = datetime(2006, 6, 30, 00, 00, 00)
                        t2 = datetime(2006, 10, 6, 00, 11, 00)
                    elif r1 == '10.2.1.6':
                        t1 = datetime(2006, 10, 6, 00, 11, 00)
                        t2 = datetime(2007, 8, 9, 00, 13, 00)
                    elif r1 == '10.3.1.4':
                        t1 = datetime(2007, 8, 9, 00, 13, 00)
                        t2 = datetime(2009, 4, 29, 18, 3, 00)
                elif project == "kafka":
                    if r1 == '0.9.0.0':
                        t1 = datetime(2015, 11, 24, 00, 00, 00)
                        t2 = datetime(2016, 5, 23, 00, 00, 00)
                    elif r1 == '0.10.0.0':
                        t1 = datetime(2016, 5, 23, 00, 00, 00)
                        t2 = datetime(2016, 10, 19, 00, 00, 00)
                    elif r1 == '0.10.1.0':
                        t1 = datetime(2016, 10, 19, 00, 00, 00)
                        t2 = datetime(2017, 2, 21, 00, 00, 00)
                    elif r1 == '0.10.2.0':
                        t1 = datetime(2017, 2, 21, 00, 00, 00)
                        t2 = datetime(2017, 6, 28, 00, 00, 00)
                elif project == "activemq":
                    if r1 == 'activemq-4.1.1':
                        t1 = datetime(2007, 4, 2, 15, 19, 00)
                        t2 = datetime(2007, 12, 13, 16, 35, 00)
                    elif r1 == 'activemq-5.0.0':
                        t1 = datetime(2007, 12, 13, 16, 35, 00)
                        t2 = datetime(2008, 5, 6, 17, 9, 00)
                    elif r1 == 'activemq-5.1.0':
                        t1 = datetime(2008, 5, 6, 17, 9, 00)
                        t2 = datetime(2008, 11, 20, 14, 51, 00)
                    elif r1 == 'activemq-5.2.0':
                        t1 = datetime(2008, 11, 20, 14, 51, 00)
                        t2 = datetime(2009, 10, 13, 9, 1, 00)
                    elif r1 == 'activemq-5.3.0':
                        t1 = datetime(2009, 10, 13, 9, 1, 00)
                        t2 = datetime(2011, 4, 1, 11, 5, 00)
                elif project == "maven":
                    if r1 == 'maven-3.0':
                        t1 = datetime(2010, 10, 8, 00, 00, 00)
                        t2 = datetime(2013, 7, 15, 00, 00, 00)
                    elif r1 == 'maven-3.1.0':
                        t1 = datetime(2013, 7, 15, 00, 00, 00)
                        t2 = datetime(2015, 11, 14, 00, 00, 00)
                    elif r1 == 'maven-3.3.9':
                        t1 = datetime(2015, 11, 14, 00, 00, 00)
                        t2 = datetime(2017, 4, 7, 00, 00, 00)
                elif project == "struts":
                    if r1 == 'STRUTS_2_3_20':
                        t1 = datetime(2014, 11, 21, 00, 00, 00)
                        t2 = datetime(2015, 5, 3, 00, 00, 00)
                    elif r1 == 'STRUTS_2_3_24':
                        t1 = datetime(2015, 5, 3, 00, 00, 00)
                        t2 = datetime(2016, 3, 18, 00, 00, 00)
                    elif r1 == 'STRUTS_2_3_28':
                        t1 = datetime(2016, 3, 18, 00, 00, 00)
                        t2 = datetime(2017, 3, 6, 00, 00, 00)
                elif project == "nifi":
                    if r1 == 'nifi-0.4.0':
                        t1 = datetime(2015, 12, 8, 00, 00, 00)
                        t2 = datetime(2016, 2, 12, 00, 00, 00)
                    elif r1 == 'nifi-0.5.0':
                        t1 = datetime(2016, 2, 12, 00, 00, 00)
                        t2 = datetime(2016, 3, 23, 00, 00, 00)
                    elif r1 == 'nifi-0.6.0':
                        t1 = datetime(2016, 3, 23, 00, 00, 00)
                        t2 = datetime(2016, 7, 12, 00, 00, 00)
                else:
                    for c10 in c1:
                        t1 = self.commit_time[c10]
                        print(f"t1 = {t1}")
                    for c20 in c2:
                        t2 = self.commit_time[c20]
                        print(f"t2 = {t2}")
                if t1 and t2:
                    if project not in self.project_release_commits:
                        self.project_release_commits[project] = defaultdict()
                    if r2 not in self.project_release_commits[project]:
                        self.project_release_commits[project][r2] = list()
                    for commit in commits:
                        if t1 <= self.commit_time[commit] < t2:
                            if commit not in self.project_release_commits[project][r2]:
                                self.project_release_commits[project][r2].append(commit)
        for project in self.project_release_commits:
            print(f"project = {project}")
            for release in self.project_release_commits[project]:
                print(f"release = {release} | commits = {len(self.project_release_commits[project][release])}")

        with open(f"project_release_commits_final_new.pkl", "wb") as f:
            pickle.dump(self.project_release_commits, f)


if __name__ == "__main__":
    st = datetime.now()
    print(f"start time : {datetime.now()}")
    t = Task2()
    t.build_commit_time()
    t.build_commit_time_tag()
    t.build_project_release_times()
    t.build_project_release_time_commit()
    print(f"start time = {st} | end time = {datetime.now()}")
