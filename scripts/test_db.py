import psycopg2
import os
from dotenv import load_dotenv

load_dotenv()

def test_connection():
    dns = os.getenv("dns")
    print(f"尝试连接到: {dns}")
    try:
        conn = psycopg2.connect(dns)
        cur = conn.cursor()
        cur.execute("SELECT version();")
        print(f"PostgreSQL 版本: {cur.fetchone()}")
        
        cur.execute("SELECT * FROM pg_extension WHERE extname = 'vector';")
        if cur.fetchone():
            print("✓ pgvector 扩展已安装")
        else:
            print("✗ pgvector 扩展未安装")
            print("正在尝试安装 pgvector...")
            cur.execute("CREATE EXTENSION IF NOT EXISTS vector;")
            conn.commit()
            print("✓ pgvector 扩展已成功创建")
            
        cur.close()
        conn.close()
        print("✓ 数据库连接与配置测试通过！")
    except Exception as e:
        print(f"✗ 连接失败: {e}")

if __name__ == "__main__":
    test_connection()
