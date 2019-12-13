from bs4 import BeautifulSoup
import re
import json


class PatentDes:
    def __init__(self, des_file):
        """
        原始文件中应该包含专利号、标题、技术领域、背景技术、发明内容、附图说明、具体实施方式字段
        但不是每个文件中都一定包含如下字段，需要判断
        """
        self.des_file = des_file
        self.soup = BeautifulSoup(open(self.des_file), 'html.parser')
        self.chinese = self.soup.chinese.get_text()

        self.number = ""
        self.title = ""
        self.technical_field = ""
        self.background_art = ""
        self.invention_content = ""
        self.drawings = ""
        self.implementation = ""

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
