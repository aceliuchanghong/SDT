# sqlite
LOG_LEVEL = "ERROR"
db_path = "summary/media_sum.db"
segment_length = 3500

create_table_sql = """
CREATE TABLE IF NOT EXISTS media_sum_info
(summaryType TEXT,filePath TEXT,text_detail TEXT, fileInfo TEXT, sumText TEXT, time_insert TEXT, remark TEXT)
"""
table_select_url_count_sql = """
select count(*) from media_sum_info where summaryType = ? and filePath = ?
"""
table_select_url_sql = """
select * from media_sum_info where summaryType = ? and filePath = ?
"""
table_select_sum_sql = """
select sumText from media_sum_info where summaryType = ? and filePath = ?
"""
table_select_text_sql = """
select text_detail from media_sum_info where summaryType = ? and filePath = ?
"""
table_count_sql = """
select count(*) from media_sum_info
"""
table_all_sql = """
select * from media_sum_info
"""
table_add_sql = """
INSERT INTO media_sum_info (summaryType, filePath, text_detail, fileInfo, sumText, time_insert, remark)
VALUES (?, ?, ?, ?, ?, ?, ?)
"""
table_del_url_sql = """
delete from media_sum_info where summaryType = ? and filePath = ?
"""
table_truncate_sql = """
DELETE FROM media_sum_info
"""
