import re

def safe_id(value):
    value=str(value).strip()
    if not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_.+-]*',value):
        raise ValueError(f'编号请用英文字母/数字，可含点、加号、下划线或短横线：{value}')
    return value

def parse_recording_name(name):
    """User-facing P01_姓名_T1.1+T1.2+T3, with legacy compatibility."""
    if '__' in name:
        parts=name.split('__')
        if len(parts)!=2:raise ValueError('旧命名仅支持被试__录制编号')
        return safe_id(parts[0]),'',safe_id(parts[1])
    parts=name.split('_',2)
    if len(parts)!=3 or not re.fullmatch(r'[Pp]\d+',parts[0]):
        raise ValueError('文件夹请命名为P01_姓名_T4或P01_姓名_T1.1+T1.2+T3；中间用单下划线，不能用斜杠')
    participant,person,recording=parts
    if not person.strip() or re.search(r'[<>:"/\\|?*]',person):raise ValueError('姓名不能为空或含路径保留符号')
    return participant.upper(),person.strip(),safe_id(recording)

def output_name(participant,person,task):
    return f'{participant}_{person}_{safe_id(task)}' if person else f'{participant}__{safe_id(task)}'
