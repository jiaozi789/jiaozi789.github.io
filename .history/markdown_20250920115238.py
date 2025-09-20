import os
import re
import requests
from urllib.parse import urlparse, quote
import logging
from pathlib import Path
import urllib3
from requests.adapters import HTTPAdapter
from requests.packages.urllib3.util.retry import Retry
import datetime  # 需要在文件顶部添加这个导入

# 禁用SSL警告
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

class MarkdownImageDownloader:
    def __init__(self):
        # 创建重试策略
        retry_strategy = Retry(
            total=3,
            backoff_factor=0.5,
            status_forcelist=[429, 500, 502, 503, 504],
            allowed_methods=["GET"]
        )
        
        # 创建适配器
        adapter = HTTPAdapter(max_retries=retry_strategy, pool_connections=10, pool_maxsize=10)
        
        # 创建会话
        self.session = requests.Session()
        
        # 移除代理设置（避免使用系统代理）
        self.session.trust_env = False
        
        # 挂载适配器
        self.session.mount("http://", adapter)
        self.session.mount("https://", adapter)
        
        # 设置请求头
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
            'Accept': 'image/webp,image/apng,image/*,*/*;q=0.8',
            'Accept-Language': 'zh-CN,zh;q=0.9,en;q=0.8',
            'Connection': 'keep-alive',
        })
        
        logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    
    def download_image(self, url, save_path):
        """下载单张图片到指定路径"""
        try:
            # 禁用SSL验证，设置超时，不验证证书
            response = self.session.get(
                url, 
                timeout=(10, 30),  # 连接超时10秒，读取超时30秒
                verify=False,      # 不验证SSL证书
                allow_redirects=True  # 允许重定向
            )
            response.raise_for_status()
            
            # 检查内容类型是否为图片
            content_type = response.headers.get('content-type', '')
            if not content_type.startswith('image/'):
                logging.warning(f"URL返回的内容不是图片: {url}, Content-Type: {content_type}")
                return False
            
            # 确保目录存在
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            
            # 保存图片
            with open(save_path, 'wb') as f:
                f.write(response.content)
            
            logging.info(f"图片下载成功: {save_path}")
            return True
            
        except requests.exceptions.RequestException as e:
            logging.error(f"网络请求失败 {url}: {str(e)}")
            return False
        except Exception as e:
            logging.error(f"下载图片失败 {url}: {str(e)}")
            return False
    
    def extract_image_urls(self, markdown_content):
        """从Markdown内容中提取所有图片URL和alt文本"""
        # 匹配标准的Markdown图片语法 ![alt](url)
        pattern = r'!\[(.*?)\]\((.*?)\)'
        matches = re.findall(pattern, markdown_content)
        
        # 返回包含alt文本和URL的元组列表
        return matches
    
    def format_markdown_path(self, path):
        """使用URL编码处理特殊字符"""
        # return "/docs/"+quote(path, safe='')
        path = path.replace("\\", "/")
        return "/docs/"+path
    
    def sanitize_filename(self, filename):
        """清理文件名，移除非法字符"""
        # 移除URL参数等
        filename = filename.split('?')[0].split('#')[0]
        
        # 替换非法字符
        invalid_chars = '<>:"/\\|?* '
        for char in invalid_chars:
            filename = filename.replace(char, '_')
        
        # 限制文件名长度
        if len(filename) > 100:
            name, ext = os.path.splitext(filename)
            filename = name[:100-len(ext)] + ext
        return filename
    
    def process_markdown_file(self, file_path, replace_links=True):
        file_path = file_path.replace("\\", "/")
        """处理Markdown文件，下载所有图片并替换链接"""
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"文件不存在: {file_path}")
        
        # 读取Markdown文件内容
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
        except UnicodeDecodeError:
            # 尝试其他编码
            with open(file_path, 'r', encoding='gbk') as f:
                content = f.read()
        
        # 提取图片URL和alt文本
        image_matches = self.extract_image_urls(content)
        if not image_matches:
            logging.info("未找到图片链接")
            return content
        
        logging.info(f"找到 {len(image_matches)} 个图片链接")
        
        # 创建图片保存目录
        base_images_dir = "static/images"  # 新增基础images目录
        replace_base_images_dir = "images"  # 新增基础images目录
        image_dir = os.path.join(base_images_dir, f"{file_path}.images")
        replace_image_dir = os.path.join(replace_base_images_dir, f"{file_path}.images")
        os.makedirs(image_dir, exist_ok=True)
        
        # 用于存储替换映射
        replacement_map = []
        
        # 下载所有图片并构建替换映射
        downloaded_count = 0
        for i, (alt_text, url) in enumerate(image_matches, 1):
            # 只处理http或https开头的网络图片（修改这里）
            if not url.strip() or not url.startswith(('http://', 'https://')):
                logging.info(f"跳过本地图片或无效URL: {url}")
                continue
            # 跳过本地路径和data URI
            if url.startswith(('data:image/', './', '/', 'file://')) or not url.strip():
                logging.info(f"跳过本地图片或空URL: {url}")
                continue
            
            logging.info(f"正在处理第 {i}/{len(image_matches)} 张图片: {url}")
            
            # 从URL中提取文件名
            parsed_url = urlparse(url)
            filename = os.path.basename(parsed_url.path)
            
            # 如果URL中没有文件名或文件名不合法，使用索引作为文件名
            if not filename or '.' not in filename:
                filename = f"image_{i}.png"
            else:
                # 清理文件名
                filename = self.sanitize_filename(filename)
            
            # 完整的保存路径
            save_path = os.path.join(image_dir, filename)
            
            # 本地相对路径（相对于Markdown文件）
            # relative_path = os.path.join(f"{os.path.basename(file_path)}.images", filename)
            relative_path = os.path.join(base_images_dir, f"{os.path.basename(file_path)}.images", filename)
            replace_relative_path = os.path.join(replace_base_images_dir, f"{file_path}.images", filename)
            
            # 格式化路径（处理空格等特殊字符）
            replace_formatted_path = self.format_markdown_path(replace_relative_path)
            
            # 如果文件已存在，跳过下载但仍然记录替换映射
            if os.path.exists(save_path):
                logging.info(f"图片已存在，跳过下载: {save_path}")
                replacement_map.append((url, replace_formatted_path, alt_text))
                downloaded_count += 1
                continue
            
            # 下载图片
            if self.download_image(url, save_path):
                replacement_map.append((url, replace_formatted_path, alt_text))
                downloaded_count += 1
        
        logging.info(f"图片处理完成: 共找到 {len(image_matches)} 张图片，成功处理 {downloaded_count} 张")
        
        # 替换Markdown中的图片链接
        if replace_links and replacement_map:
            new_content = content
            for original_url, local_path, alt_text in replacement_map:
                # 构建新的Markdown图片语法
                new_image_syntax = f"![{alt_text}]({local_path})"
                
                # 替换原始内容中的图片链接
                old_image_syntax = f"![{alt_text}]({original_url})"
                new_content = new_content.replace(old_image_syntax, new_image_syntax)
            
            # 写回文件
            # 创建备份目录和带日期的备份文件（修改这里）
            backup_dir = "static/baks"
            os.makedirs(backup_dir, exist_ok=True)
            # 生成带日期时间的备份文件名
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            file_basename = os.path.basename(file_path)
            backup_filename = f"{file_basename}.{timestamp}.bak"
            backup_path = os.path.join(backup_dir, backup_filename)
            with open(backup_path, 'w', encoding='utf-8') as f:
                f.write(content)
            logging.info(f"已创建备份文件: {backup_path}")
            
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(new_content)
            logging.info(f"已更新Markdown文件中的图片链接")
            
            return new_content
        
        return content

if __name__ == "__main__":
    downloader = MarkdownImageDownloader()
    
    # 指定要扫描的目录
    scan_directory = "content"  # 当前目录，可以修改为其他路径
    
    # 递归扫描所有md文件，排除baks目录
    md_files = []
    for root, dirs, files in os.walk(scan_directory):
        # 排除baks目录及其子目录
        if 'baks' in dirs:
            dirs.remove('baks')
        
        for file in files:
            if file.endswith('.md'):
                full_path = os.path.join(root, file)
                md_files.append(full_path)
    
    print(f"找到 {len(md_files)} 个Markdown文件")
    
    md_file=["content/devops/networking/basic/net_basic_01.md"]
    # 处理所有找到的md文件
    processed_count = 0
    for md_file in md_files:
        try:
            print(f"\n正在处理: {md_file}")
            # 处理文件并替换链接
            new_content = downloader.process_markdown_file(md_file, replace_links=True)
            processed_count += 1
            print(f"处理完成: {md_file}")
            
        except Exception as e:
            print(f"处理失败 {md_file}: {str(e)}")
    
    print(f"\n处理完成！共处理 {processed_count}/{len(md_files)} 个文件")