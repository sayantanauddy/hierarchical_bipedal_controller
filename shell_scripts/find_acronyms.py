import subprocess
import shlex
import re
import os

home_dir = os.path.expanduser('~')

# Set the directory for saving plots
report_dir = os.path.join(home_dir, 'computing/repositories/MScThesis_SayantanAuddy_2017_NICOOscillatorWalking/report')

FOLDER = report_dir
THESIS = os.path.join(report_dir,'thesis.pdf')
OUTPUT_FILE = os.path.join(report_dir,'acronymsInMyThesis.txt')
PATTERN = '[A-Z][A-Z]+'

def searchAcronymsInPDF():
    output = pdfSearch()
    acrs = []
    for reg in re.findall(PATTERN, output):
        reg.strip()
        if (len(reg)>1):
            acrs.append(reg)
    return set(acrs)

def pdfSearch():
    command = 'pdfgrep "%s" %s'%(PATTERN,THESIS)
    output = shellCall(command)
    return output

def shellCall(command):
    p = subprocess.Popen(shlex.split(command), stdout=subprocess.PIPE)
    out, _ = p.communicate()
    return out

if __name__ == '__main__':
    acrs = searchAcronymsInPDF()
    acrs_list = list(acrs)
    print(sorted(acrs_list))