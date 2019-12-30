import bs4
from bs4 import BeautifulSoup
import re


class PatentDes:
    def __init__(self, des_file):
        """
        原始文件中应该包含专利号、标题、技术领域、背景技术、发明内容、附图说明、具体实施方式字段
        但不是每个文件中都一定包含如下字段，需要判断
        """
        self.des_file = des_file
        self.soup = BeautifulSoup(open(self.des_file), 'html.parser')
        self.chinese = self.soup.chinese
        self.chinese_text = self._math_process()

        self.technical_field_keys = ["技术领域", "发明领域"]
        self.background_art_keys = ["背景技术", "技术背景", "发明背景", "现有技术", "背景"]
        self.invention_content_keys = ["发明内容", "本发明的内容"]
        self.drawings_keys = ["附图说明"]
        self.implementation_keys = ["具体实施方式", "具体实施方案"]
        self.special_char = ["【", "】", "[", "]"]

        self.number = self._get_number()
        self.title = self._get_title()
        self.technical_field = self._get_field(self.technical_field_keys, self.background_art_keys, self.invention_content_keys, self.drawings_keys,
                                               self.implementation_keys)
        self.background_art = self._get_field(self.background_art_keys, self.invention_content_keys, self.drawings_keys, self.implementation_keys)
        self.invention_content = self._get_field(self.invention_content_keys, self.drawings_keys, self.implementation_keys)
        self.drawings = self._get_field(self.drawings_keys, self.implementation_keys)
        self.implementation = self._get_field(self.implementation_keys)

    def _math_process(self):
        """
        处理其中的数学公式
        """
        subs = self.chinese.find_all("sub")
        for sub in subs:
            sub.string.replace_with("@@_(" + sub.string.strip() + ")@@")

        sups = self.chinese.find_all("sub")
        for sup in sups:
            sup.string.replace_with("@@^(" + sup.string.strip() + ")@@")

        maths = self.chinese.find_all("maths")
        for math in maths:
            for content in math.descendants:
                if isinstance(content, bs4.element.Tag) and content.string is not None:
                    content.string.replace_with(content.string.strip())
                    if content.string != "":
                        content.string = content.string + "@@"

        chinese_text = "\n".join(self.chinese.stripped_strings).replace("\n@@", "").replace("@@\n", "")
        return chinese_text

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
        pattern += ")"
        return pattern

    def _get_number(self):
        """
        获取专利号
        """
        return self.soup.title.string.strip()[2:]

    def _get_title(self):
        """
        获取专利标题
        """
        title = self.chinese.split()[0]
        title = re.sub(r" +", ' ', title)
        return title

    def _get_field(self, *keywords_list):
        """
        根据匹配的关键词获取说明书中的不同部分
        """
        field = ""
        pattern = self._get_pattern(keywords_list)
        re_field = re.search(pattern, self.chinese_text, re.DOTALL)

        if re_field is None:
            return pattern

        field = re_field.group()
        match_keys = re_field.groups()

        replace_pattern = "(" + "|".join(match_keys) + "|".join(self.special_char) + ")"
        field = re.sub(replace_pattern, "", field).strip()

        return field

    def get_json(self):
        """
        生成json结构文件
        """
        des_json = {}
        des_json["number"] = self.number
        des_json["title"] = self.title
        des_json["technical_field"] = self.technical_field
        des_json["background_art"] = self.background_art
        des_json["invention_content"] = self.invention_content
        des_json["drawings"] = self.drawings
        des_json["implementation"] = self.implementation

        return des_json


class PatentClaim:
    """
    权利要求文件中包含专利号，独立要求和从属要求
    其中独立要求一定有，从属要求不一定有
    """

    def __init__(self, claim_file):
        self.claim_file = claim_file
        self.soup = BeautifulSoup(open(self.claim_file), 'html.parser')
        self.chinese = self.soup.chinese
        self._math_process()
        self.claims = self.chinese.find_all("p")

        self.keys = ["特征在于", "特征是", "包括", "其中", "其特长"]
        self.pattern = "[" + "|".join(self.keys) + "]"

        self.number = self._get_number()
        self.independent = []
        self._get_independent()
        self.dependent = []
        self._get_dependent()

    def _math_process(self):
        """
        处理其中的数学公式
        """
        subs = self.chinese.find_all("sub")
        for sub in subs:
            sub.string.replace_with("@@_(" + sub.string.strip() + ")@@")

        sups = self.chinese.find_all("sup")
        for sup in sups:
            sup.string.replace_with("@@^(" + sup.string.strip() + ")@@")

        maths = self.chinese.find_all("maths")
        for math in maths:
            for content in math.descendants:
                if isinstance(content, bs4.element.Tag) and content.string is not None:
                    content.string.replace_with(content.string.strip())
                    if content.string != "":
                        content.string = content.string + "@@"

    def _get_number(self):
        """
        获取专利号
        """
        return self.title.string.strip()[2:]

    def _get_independent(self):
        """
        获取独立权利要求
        """
        for claim in self.claims:
            if claim["claim"] == "independent":
                preamble = ""
                characterizing = ""
                content = "\n".join(claim.stripped_strings).replace_with("\n@@", "").replace_with("@@\n", "")
                independent = re.split(self.pattern, content, maxsplit=1, flags=re.DOTALL)
                if len(independent) == 2:
                    preamble, characterizing = independent[0], independent[1]
                elif len(independent) == 1:
                    preamble = independent[0]
                self.independent.append([preamble, characterizing])

    def _get_dependent(self):
        """
        获取从属权利要求
        """
        for claim in self.claims:
            if claim["claim"] == "dependent":
                preamble = ""
                characterizing = ""
                content = "\n".join(claim.stripped_strings).replace_with("\n@@", "").replace_with("@@\n", "")
                dependent = re.split(self.pattern, content, maxsplit=1, flags=re.DOTALL)
                if len(dependent) == 2:
                    preamble, characterizing = dependent[0], dependent[1]
                elif len(dependent) == 1:
                    preamble = dependent[0]
                self.dependent.append([preamble, characterizing])


    def get_json(self):
        """
        生成json结构文件
        """
        claim_json = {}
        claim_json["number"] = self.number
        claim_json["independent"] = self.independent
        claim_json["dependent"] = self.dependent

