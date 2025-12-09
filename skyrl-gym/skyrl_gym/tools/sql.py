from skyrl_gym.tools.core import tool, ToolGroup
import pandas as pd
import sqlite3
import sys
import concurrent.futures
import os


class SQLCodeExecutorToolGroup(ToolGroup):
    def __init__(self, db_file_path: str):
        self.db_path = db_file_path
        super().__init__(name="SQLCodeExecutorToolGroup")

    @tool
    def sql(self, db_id, sql, turns_left, timeout=5) -> str:
        def _execute_sql(db_file, conn: sqlite3.Connection, cursor: sqlite3.Cursor, sql: str) -> frozenset:
            try:
                conn.execute("BEGIN TRANSACTION;")
                cursor.execute(sql)
                execution_res = frozenset(cursor.fetchall())
                conn.rollback()
                return execution_res
            except Exception as e:
                conn.rollback()
                return f"Error executing SQL: {str(e)}, db file: {db_file}"

        def _execute_sql_wrapper(db_file, sql, timeout=5) -> str:
            conn = None
            try:
                conn = sqlite3.connect(db_file)
                cursor = conn.cursor()
                with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
                    future = executor.submit(_execute_sql, db_file, conn, cursor, sql)
                    try:
                        res = future.result(timeout=timeout)
                    except concurrent.futures.TimeoutError:
                        conn.rollback()
                        return f"SQL Timeout:\n{sql}"
                    except Exception as e:
                        conn.rollback()
                        conn.close()
                        return f"Error executing SQL: {str(e)}, db file: {db_file}"

                if isinstance(res, frozenset):
                    df = pd.DataFrame(res)
                    res = df.to_string(index=False)
                    # NOTE: observation too long, just truncate
                    if len(res) > 9000:
                        # just truncate
                        truncated_df = df.head(50)
                        res = "Truncated to 50 lines since returned response too long: " + truncated_df.to_string(
                            index=False
                        )  # or index=True if you want row numbers
                else:
                    res = str(res)

            except KeyboardInterrupt:
                sys.exit(0)
            except Exception as e:
                res = str(e)
            finally:
                # Always close the connection to ensure cleanup of temporary files
                if conn is not None:
                    try:
                        conn.close()
                    except Exception:
                        pass  # Ignore errors during cleanup

            return res

        # TODO (erictang000): move this logic up into the text2sql env, since this is more specific logic
        reminder_text = f"<reminder>You have {turns_left} turns left to complete the task.</reminder>"
        if sql is None:
            obs = "Your previous action is invalid. Follow the format of outputting thinking process and sql tool, and try again."
        else:
            db_file = os.path.join(self.db_path, db_id, db_id + ".sqlite")
            obs = _execute_sql_wrapper(db_file, sql, timeout)

        return f"\n\n<observation>{obs}\n{reminder_text}</observation>\n\n"
