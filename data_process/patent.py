import re
import json
import bs4
from bs4 import BeautifulSoup


class Patent:
    def __init__(self, des_file, claim_file, lcs_threshold):
        """
        说明书中中应该包含专利号、标题、技术领域、背景技术、发明内容、附图说明、具体实施方式字段
        但不是必须包含以下字段，需要根据正则表达式判断
        根据正则表达式匹配
        """
        self.des_file = des_file
        self.claim_file = claim_file
        self.lcs_threshold = lcs_threshold

        # 处理说明书部分，获取相应字段
        self.des_soup = BeautifulSoup(open(self.des_file, errors="ignore"), "html.parser")
        if self.des_soup.chinese is None:
            print(des_file)
        self.des_chinese = self._math_process(self.des_soup.chinese)
        self.des_chinese_text = "\n".join(self.des_chinese.stripped_strings).replace("\n@@", "").replace("@@\n", "")

        self.technical_field_keys = ["技术领域", "发明领域", "技术或应用领域"]
        self.background_art_keys = ["背景技术", "技术背景", "发明背景", "现有技术", "现有技术背景", "背景"]
        self.invention_content_keys = ["发明内容", "本发明的内容", "实用新型内容", "本实用新型的内容"]
        self.drawings_keys = ["附图说明"]
        self.implementation_keys = ["具体实施方式", "具体实施方案", "具体实施例", "本发明实施例"]
        self.special_chars = ["【", "】", "[", "]"]

        self.number = self._get_number()
        self.title = self._get_title()
        self.technical_field = self._get_field(self.technical_field_keys, self.background_art_keys, self.invention_content_keys, self.drawings_keys,
                                               self.implementation_keys)
        self.background_art = self._get_field(self.background_art_keys, self.invention_content_keys, self.drawings_keys, self.implementation_keys)
        self.invention_content = self._get_field(self.invention_content_keys, self.drawings_keys, self.implementation_keys)
        self.drawings = self._get_field(self.drawings_keys, self.implementation_keys)
        self.implementation = self._get_field(self.implementation_keys)

        # 处理权利要求书部分，获取相应字段
        self.claim_soup = BeautifulSoup(open(self.claim_file, errors="ignore"), "html.parser")
        if self.claim_soup.chinese is None:
            print(claim_file)
        self.claim_chinese = self._math_process(self.claim_soup.chinese)
        self.claims = self.claim_chinese.find_all("p")

        self.claim_keys = ["其特征在于", "特征在于", "其特征是", "特征是", "包括", "其中", "其特长", "流程如下", "其步骤为", "步骤中", "对象为"]

        self.independent = self._get_independent()
        self.dependent = self._get_dependent()
        if self.invention_content != "":
            self._align()

    def _math_process(self, chinese):
        """
        处理其中的数学公式
        """
        us = chinese.find_all("u")
        for u in us:
            if u.string is None:
                for content in u.descendants:
                    if isinstance(content, bs4.element.Tag) and content.string is not None:
                        content.string.replace_with(content.string.strip())
                        if content.string != "":
                            content.string = "_" + content.string + "_"
            else:
                u.string.replace_with("_" + u.string.strip() + "_")

        subs = chinese.find_all("sub")
        for sub in subs:
            if sub.string is None:
                for content in sub.descendants:
                    if isinstance(content, bs4.element.Tag) and content.string is not None:
                        content.string.replace_with(content.string.strip())
                        if content.string != "":
                            content.string = "@@_(" + content.string + ")@@"
            else:
                sub.string.replace_with("@@_(" + sub.string.strip() + ")@@")

        sups = chinese.find_all("sup")
        for sup in sups:
            if sup.string is None:
                for content in sup.descendants:
                    if isinstance(content, bs4.element.Tag) and content.string is not None:
                        content.string.replace_with(content.string.strip())
                        if content.string != "":
                            content.string = "@@^(" + content.string + ")@@"
            else:
                sup.string.replace_with("@@^(" + sup.string.strip() + ")@@")

        maths = chinese.find_all("maths")
        for math in maths:
            for content in math.descendants:
                if isinstance(content, bs4.element.Tag) and content.string is not None:
                    content.string.replace_with(content.string.strip())
                    if content.string != "":
                        content.string = content.string + "@@"

        return chinese

    def _get_pattern(self, *keywords_list):
        """
        获取正则表达式的pattern
        """
        assert len(keywords_list) >= 1

        if len(keywords_list) == 1:
            return "(" + "|".join(keywords_list[0]) + ")" + ".*"

        pattern = "(" + "|".join(keywords_list[0]) + ")" + ".*?" + "("
        for keywords in keywords_list[1:]:
            pattern += "|".join(keywords)
            if len(keywords) == 1:
                pattern += "|"
        pattern += ")"
        return pattern

    def _get_number(self):
        """
        获取专利号
        """
        assert self.des_soup.title.string is not None
        return self.des_soup.title.string.strip()[2:]

    def _get_title(self):
        """
        获取专利标题
        """
        title = self.des_chinese_text.split()[0]
        return title

    def _get_field(self, *keywords_list):
        """
        根据匹配的关键词获取说明书中的不同部分
        """
        field = ""
        pattern = self._get_pattern(*keywords_list)
        re_field = re.search(pattern, self.des_chinese_text, re.DOTALL)

        if re_field is None:
            return field

        field = re_field.group()
        match_keys = re_field.groups()

        replace_pattern = "(" + "|".join(match_keys) + "|" + "|".join(self.special_chars) + ")"
        field = re.sub(replace_pattern, "", field.strip())

        return field

    def _get_independent(self):
        """
        获取独立权利要求
        """
        independent = list()

        sentence_split = re.compile(r"[,;，；：！？｡。!?:]")
        for claim in self.claims:
            if claim["claim"] == "independent":
                preamble = ""
                characterizing = ""
                content = "\n".join(claim.stripped_strings).replace("\n@@", "").replace("@@\n", "")
                for key in self.claim_keys:
                    if key in content:
                        independ = content.split(key, 1)
                        preamble, characterizing = independ[0], independ[1]
                        break
                if preamble == "" and characterizing == "":
                    if "：" in content and len(sentence_split.findall(content.split("：", 1)[0])) < 3:
                        preamble = content.split("：", 1)[0]
                        characterizing = content.split("：", 1)[1]
                    elif ":" in content and len(sentence_split.findall(content.split(":", 1)[0])) < 3:
                        preamble = content.split(":", 1)[0]
                        characterizing = content.split(":", 1)[1]
                    else:
                        characterizing = content
                independent.append([preamble, characterizing])

        return independent

    def _get_dependent(self):
        """
        获取从属权利要求
        """
        dependent = list()

        sentence_split = re.compile(r"[,;，；：！？｡。!?:]")
        for claim in self.claims:
            if claim["claim"] == "dependent":
                reference = ""
                limited = ""
                content = "\n".join(claim.stripped_strings).replace("\n@@", "").replace("@@\n", "")
                for key in self.claim_keys:
                    if key in content:
                        depend = content.split(key, 1)
                        reference, limited = depend[0], depend[1]
                        break
                if reference == "" and limited == "":
                    if "：" in content and len(sentence_split.findall(content.split("：", 1)[0])) < 3:
                        reference = content.split("：", 1)[0]
                        limited = content.split("：", 1)[1]
                    elif ":" in content and len(sentence_split.findall(content.split(":", 1)[0])) < 3:
                        reference = content.split(":", 1)[0]
                        limited = content.split(":", 1)[1]
                    else:
                        limited = content
                dependent.append([reference, limited])
        return dependent

    def _cut_sents(self, text):
        """
        断句，如果双引号前有终止符，那么双引号才是句子的终点，把分句符放在双引号后
        """
        total_len = len(text)
        para = text

        para = re.sub(r'([：；，､\u3000、﹔·！？｡。])([^”’])', r"\1@@sub@@\2", para)  # 单字符断句符
        para = re.sub(r'(\.{6})([^”’])', r"\1@@sub@@\2", para)  # 英文省略号
        para = re.sub(r'(…{2})([^”’])', r"\1@@sub@@\2", para)  # 中文省略号
        para = re.sub(r'([：；，､\u3000、﹔·！？｡。][”’])([^，。！？?])', r'\1@@sub@@\2', para)

        span = list()
        prev = 0

        index = para.find("@@sub@@")
        while index > 0:
            if index > 3:
                span.append((prev, prev + index))
            prev += index
            para = para[index + 7:]
            index = para.find("@@sub@@")
        if len(para) > 3:
            span.append((prev, prev + len(para)))

        assert len(span) == 0 or span[-1][-1] <= total_len

        return span

    def _lcs(self, str_a, str_b):
        """
        寻找两个字符串间的最长公共子序列
        """
        if len(str_a) == 0 or len(str_b) == 0:
            return 0

        dp = [[0 for _ in range(len(str_b) + 1)] for _ in range(len(str_a) + 1)]
        for i in range(1, len(str_a) + 1):
            for j in range(1, len(str_b) + 1):
                if str_a[i - 1] == str_b[j - 1]:
                    dp[i][j] = dp[i - 1][j - 1] + 1
                else:
                    dp[i][j] = max([dp[i - 1][j], dp[i][j - 1]])

        return dp[len(str_a)][len(str_b)]

    def _align_cal(self, des_spans, claim_spans, claim_str):
        matched_range = (-1, -1)
        if len(des_spans) == 0 or len(claim_spans) == 0:
            return matched_range

        # 计算一个文书内文本分段间的lcs匹配率,lcs_rates中保存匹配矩阵
        lcs_rates = list()
        for i in range(len(claim_spans)):
            lcs_rates.append([])
            for j in range(len(des_spans)):
                des_span = self.invention_content[des_spans[j][0]:des_spans[j][1]]
                claim_span = claim_str[claim_spans[i][0]:claim_spans[i][1]]
                ij_lcs = self._lcs(des_span, claim_span)
                l = min(len(des_span), len(claim_span))
                assert l > 0
                lcs_rate = ij_lcs / l
                if lcs_rate >= self.lcs_threshold:
                    lcs_rates[i].append(1)
                else:
                    lcs_rates[i].append(-1)
        # 从后到前寻找最长匹配，贪心方案
        matched_des = list()
        border = len(des_spans) - 1
        for i in range(len(claim_spans) - 1, -1, -1):
            for j in range(border - 1, -1, -1):
                if lcs_rates[i][j] > 0:
                    matched_des.append((i, j))
                    border = j - 1
                    break
        if len(matched_des) > 0:
            matched_range = (des_spans[matched_des[-1][1]][0], des_spans[matched_des[0][1]][1])
        return matched_range

    def _align(self):
        des_spans = self._cut_sents(self.invention_content)

        for independ in self.independent:
            claim_str = independ[1]
            claim_spans = self._cut_sents(claim_str)
            matched_range = self._align_cal(des_spans, claim_spans, claim_str)
            independ.append(matched_range)
        for depend in self.dependent:
            claim_str = depend[1]
            claim_spans = self._cut_sents(claim_str)
            matched_range = self._align_cal(des_spans, claim_spans, claim_str)
            depend.append(matched_range)

    def get_json(self):
        """
        生成json结构文件
        """
        patent_json = dict()
        patent_json["number"] = self.number
        patent_json["title"] = self.title
        patent_json["technical_field"] = self.technical_field
        patent_json["background_art"] = self.background_art
        patent_json["invention_content"] = self.invention_content
        patent_json["drawings"] = self.drawings
        patent_json["implementation"] = self.implementation
        patent_json["independent"] = self.independent
        patent_json["dependent"] = self.dependent

        return patent_json
