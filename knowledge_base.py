import tkinter as tk
from tkinter import ttk, messagebox
import pathlib
import pandas as pd
import os
import sys
import threading
from datetime import datetime

# 引入繪圖相關庫
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

class RAGApp:
    def __init__(self, root):
        self.root = root
        self.root.title("璞真RAG 管理系統")
        self.root.geometry("1200x900")

        # 設定關閉視窗的協議，防止終端機卡住
        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)

        # 設定路徑
        self.target_folder_path = "D:/璞真RAG資料夾"
        self.list_save_path = 'file_list.csv'
        self.log_save_path = 'change_log.csv'

        # 儲存 Load Data 頁面勾選的檔案路徑
        self.selected_files = set()

        # 設定繪圖字型 (針對繁體中文環境)
        plt.rcParams['font.sans-serif'] = ['Microsoft JhengHei', 'Arial Unicode MS', 'SimHei'] 
        plt.rcParams['axes.unicode_minus'] = False

        self.setup_ui()
        
        # 程式啟動 100 毫秒後，自動執行第一次掃描
        self.root.after(100, self.start_thread_scan)

    def on_closing(self):
        """處理視窗關閉邏輯"""
        plt.close('all')
        self.root.quit()
        self.root.destroy()
        sys.exit(0)

    def setup_ui(self):
        self.notebook = ttk.Notebook(self.root)
        self.notebook.pack(expand=True, fill="both", padx=10, pady=10)

        # 四個主要分頁
        self.tab_dashboard = ttk.Frame(self.notebook)
        self.tab_load_data = ttk.Frame(self.notebook)
        self.tab_chunking = ttk.Frame(self.notebook)
        self.tab_database = ttk.Frame(self.notebook)

        self.notebook.add(self.tab_dashboard, text="Dashboard")
        self.notebook.add(self.tab_load_data, text="Load Data")
        self.notebook.add(self.tab_chunking, text="Chunking")
        self.notebook.add(self.tab_database, text="Database")

        self.setup_dashboard()
        self.setup_load_data()

    # ================= Dashboard 頁面邏輯 =================

    def setup_dashboard(self):
        # 控制按鈕
        top_btn_frame = ttk.Frame(self.tab_dashboard)
        top_btn_frame.pack(side=tk.TOP, fill="x", padx=10, pady=5)

        self.btn_scan = ttk.Button(top_btn_frame, text="掃描全部 (手動刷新)", command=self.start_thread_scan)
        self.btn_scan.pack(side=tk.LEFT, padx=5)

        # 上方：圖表視覺化
        self.chart_frame = ttk.LabelFrame(self.tab_dashboard, text="數據視覺化")
        self.chart_frame.pack(side=tk.TOP, fill="both", expand=True, padx=10, pady=5)
        
        self.fig, (self.ax1, self.ax2) = plt.subplots(1, 2, figsize=(10, 4))
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.chart_frame)
        self.canvas.get_tk_widget().pack(expand=True, fill="both")

        # 下方：掃描日誌
        self.log_frame = ttk.LabelFrame(self.tab_dashboard, text="檔案掃描日誌")
        self.log_frame.pack(side=tk.BOTTOM, fill="both", expand=True, padx=10, pady=5)
        
        self.txt_log = tk.Text(self.log_frame, height=10, font=("Consolas", 10))
        self.txt_log.pack(side=tk.LEFT, fill="both", expand=True)
        
        scrollbar = ttk.Scrollbar(self.log_frame, command=self.txt_log.yview)
        scrollbar.pack(side=tk.RIGHT, fill="y")
        self.txt_log.config(yscrollcommand=scrollbar.set)

    # ================= Load Data 頁面邏輯 =================

    def setup_load_data(self):
        ctrl_frame = ttk.Frame(self.tab_load_data)
        ctrl_frame.pack(side=tk.TOP, fill="x", padx=10, pady=5)

        btn_refresh = ttk.Button(ctrl_frame, text="重新整理清單", command=self.refresh_load_data_tree)
        btn_refresh.pack(side=tk.LEFT, padx=5)

        btn_action = ttk.Button(ctrl_frame, text="處理勾選檔案", command=self.process_selected_files)
        btn_action.pack(side=tk.LEFT, padx=5)

        # Treeview 結構
        tree_frame = ttk.Frame(self.tab_load_data)
        tree_frame.pack(expand=True, fill="both", padx=10, pady=5)

        columns = ("select", "status", "size")
        self.tree = ttk.Treeview(tree_frame, columns=columns, show="tree headings")
        
        self.tree.heading("#0", text="資料夾結構 (小三角形收合)", anchor="w")
        self.tree.heading("select", text="選取")
        self.tree.heading("status", text="處理狀態")
        self.tree.heading("size", text="大小(KB)")

        self.tree.column("#0", width=500)
        self.tree.column("select", width=60, anchor="center")
        self.tree.column("status", width=120, anchor="center")
        self.tree.column("size", width=100, anchor="e")

        # 定義背景顏色 Tags
        self.tree.tag_configure("unprocessed", background="#FFFFE0") # 淺黃
        self.tree.tag_configure("processed", background="#E0FFE0")   # 淺綠

        ysb = ttk.Scrollbar(tree_frame, orient="vertical", command=self.tree.yview)
        self.tree.configure(yscroll=ysb.set)
        
        self.tree.pack(side=tk.LEFT, expand=True, fill="both")
        ysb.pack(side=tk.RIGHT, fill="y")

        # 綁定點擊事件實現勾選框
        self.tree.bind("<ButtonRelease-1>", self.on_tree_click)

    def refresh_load_data_tree(self):
        """讀取 CSV 並構建 Treeview 階層"""
        for item in self.tree.get_children():
            self.tree.delete(item)
        
        if not os.path.exists(self.list_save_path): return
        df = pd.read_csv(self.list_save_path)
        
        folder_nodes = {}
        for _, row in df.iterrows():
            rel_path = pathlib.Path(row['相對路徑'])
            parts = rel_path.parts
            parent = ""
            # 建立父資料夾節點
            for i in range(len(parts) - 1):
                path_key = "/".join(parts[:i+1])
                if path_key not in folder_nodes:
                    node_id = self.tree.insert(parent, "end", text=parts[i], open=True)
                    folder_nodes[path_key] = node_id
                parent = folder_nodes[path_key]

            # 插入檔案
            tag = "unprocessed" if row['處理狀態'] == "unprocessed" else "processed"
            self.tree.insert(
                parent, "end", 
                text=row['檔案名稱'], 
                values=("☐", row['處理狀態'], row['檔案大小(KB)']),
                tags=(tag,),
                iid=row['檔案路徑']
            )

    def on_tree_click(self, event):
        """處理點擊事件，確保勾選與資料集合完全同步"""
        region = self.tree.identify_region(event.x, event.y)
        
        # 只有點擊到單元格 (cell) 才處理
        if region == "cell":
            column = self.tree.identify_column(event.x)
            item_id = self.tree.identify_row(event.y)
            
            # 判斷是否點擊第一欄 (選取欄 #1)
            if column == "#1": 
                # 取得該列目前的 values，轉為 list 方便修改
                vals = list(self.tree.item(item_id, "values"))
                current_icon = str(vals[0]).strip() # 去除可能存在的空白

                if current_icon == "☐":
                    # 執行勾選
                    vals[0] = "☑"
                    self.selected_files.add(item_id)
                else:
                    # 執行取消勾選
                    vals[0] = "☐"
                    # 使用 discard 即使 key 不存在也不會報錯，比 remove 安全
                    self.selected_files.discard(item_id)
                
                # 更新 Treeview 顯示
                self.tree.item(item_id, values=tuple(vals))

    def process_selected_files(self):
        if not self.selected_files:
            messagebox.showinfo("提示", "未勾選任何檔案")
            return
        messagebox.showinfo("執行", f"已選取 {len(self.selected_files)} 個檔案，準備進入後續處理。")

    # ================= 核心功能與輔助 =================

    def log_message(self, message):
        if self.root.winfo_exists():
            self.txt_log.insert(tk.END, message + "\n")
            self.txt_log.see(tk.END)
            self.root.update_idletasks()

    def start_thread_scan(self):
        self.btn_scan.config(state=tk.DISABLED)
        threading.Thread(target=self.run_scan_process, daemon=True).start()

    def get_document_tag(self, extension):
        ext = extension.lower()
        tag_map = {
            '.pdf': 'PDF 文件', '.docx': 'Word 格式', '.doc': 'Word 格式',
            '.txt': '純文字', '.md': 'Markdown 筆記', '.pptx': 'PowerPoint 簡報',
            '.csv': 'CSV', '.xlsx': 'Excel', '.xls': 'Excel',
            '.json': 'JSON', '.html': '網頁', '.jpg': '圖片', '.png': '圖片'
        }
        return tag_map.get(ext, '其他')

    def run_scan_process(self):
        try:
            self.txt_log.delete(1.0, tk.END)
            current_reading_list = []
            old_df_dict = {}
            if os.path.exists(self.list_save_path):
                try:
                    df_old = pd.read_csv(self.list_save_path)
                    old_df_dict = df_old.set_index('檔案路徑').to_dict('index')
                except: pass

            def scan_tree(current_path, root_path, depth=0):
                if not self.root.winfo_exists(): return
                try:
                    items = sorted(list(current_path.iterdir()), key=lambda x: (x.is_file(), x.name.lower()))
                    for item in items:
                        spacer = '    ' * depth
                        if item.is_dir():
                            self.log_message(f"{spacer}📁 {item.name}/")
                            scan_tree(item, root_path, depth + 1)
                        else:
                            self.log_message(f"{spacer}📄 {item.name}")
                            mtime = datetime.fromtimestamp(item.stat().st_mtime).strftime('%Y-%m-%d %H:%M:%S')
                            file_path = str(item.resolve())
                            status = 'unprocessed'
                            if file_path in old_df_dict:
                                if old_df_dict[file_path].get('最後修改時間') == mtime:
                                    status = old_df_dict[file_path].get('處理狀態', 'unprocessed')
                            
                            current_reading_list.append({
                                '檔案名稱': item.name, '文件種類標籤': self.get_document_tag(item.suffix),
                                '檔案路徑': file_path, '相對路徑': str(item.relative_to(root_path)),
                                '副檔名': item.suffix, '檔案大小(KB)': round(item.stat().st_size / 1024, 2),
                                '最後修改時間': mtime, '處理狀態': status
                            })
                except: pass

            root_dir = pathlib.Path(self.target_folder_path)
            if root_dir.exists():
                scan_tree(root_dir, root_dir)
                df = pd.DataFrame(current_reading_list)
                if not df.empty:
                    df.to_csv(self.list_save_path, index=False, encoding='utf-8-sig')
                    self.update_visuals()
                    self.refresh_load_data_tree() # 更新 Load Data 清單
                    self.log_message(f"\n✅ 掃描完成。待處理: {len(df[df['處理狀態']=='unprocessed'])}")
            
            self.btn_scan.config(state=tk.NORMAL)
        except Exception as e: print(e)

    def update_visuals(self):
        if not os.path.exists(self.list_save_path): return
        df = pd.read_csv(self.list_save_path)
        if df.empty: return

        type_stats = df.groupby('文件種類標籤').agg(數量=('檔案名稱', 'count'), 大小_KB=('檔案大小(KB)', 'sum')).reset_index()
        self.ax1.clear(); self.ax2.clear()
        colors = sns.color_palette('pastel')[0:len(type_stats)]

        self.ax1.pie(type_stats['數量'], labels=type_stats['文件種類標籤'], autopct='%1.1f%%', startangle=140, colors=colors)
        self.ax1.set_title(f"檔案數量 (共 {len(df)} 個)")
        self.ax2.pie(type_stats['大小_KB'], labels=type_stats['文件種類標籤'], autopct='%1.1f%%', startangle=140, colors=colors)
        self.ax2.set_title(f"占用空間 (共 {df['檔案大小(KB)'].sum()/1024:.2f} MB)")
        
        self.fig.tight_layout()
        self.canvas.draw()

if __name__ == "__main__":
    try:
        root = tk.Tk()
        app = RAGApp(root)
        root.mainloop()
    except KeyboardInterrupt:
        sys.exit(0)