from pathlib import Path
from g_analysis.reporting import generate_reports
if __name__=='__main__':
    generate_reports(Path(__file__).resolve().parent)
    print('完成：各任务report_figures、output/group_reports/coverage_regions.csv；截图为场景示意。')
