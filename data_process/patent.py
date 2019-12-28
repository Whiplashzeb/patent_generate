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

        self.number = ""
        self.title = ""
        self.technical_field = ""
        self.background_art = ""
        self.invention_content = ""
        self.drawings = ""
        self.implementation = ""

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

    def _get_number(self):
        """
        获取专利号
        """
        self.number = self.soup.title.string.strip()

    def _get_title(self):
        """
        获取专利标题
        """
        self.title = self.chinese.split()[0]
        self.title = re.sub(r" +", ' ', self.title)

    def _get_technical_field(self):
        """
        获取技术领域
        """
        technical_field = list()
        technical = ""
        if self.chinese.find("技术背景") != -1:
            technical_field = re.findall("[技术领域|发明领域].*技术背景", self.chinese, re.DOTALL)
        elif self.chinese.find("背景技术") != -1:
            technical_field = re.findall("[技术领域|]")
