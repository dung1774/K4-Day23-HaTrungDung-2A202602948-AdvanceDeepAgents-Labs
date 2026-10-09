# Advanced Deep Agents – Lab Day 23

Repository xây dựng hệ thống **Deep Research Multi-Agent** cho VinUni AI20K. Người dùng nhập một chủ đề; hệ thống lập kế hoạch, giao các câu hỏi con cho researcher, thu thập nguồn thật từ arXiv, Hugging Face và web qua Exa, rồi tạo báo cáo tiếng Anh có trích dẫn được kiểm tra tự động.

## Kiến trúc

- **Lead Agent**: lập kế hoạch bằng `write_todos`, giao tối thiểu ba nhiệm vụ nghiên cứu, tổng hợp theo chủ đề, tạo `sources.json`, chạy finalizer và validator trong sandbox.
- **Researcher**: dùng `arxiv_search`, `hf_daily_papers`, `hf_search_papers`, `web_search`, `web_fetch`; mỗi câu hỏi dùng ít nhất hai họ nguồn và ghi evidence vào sandbox.
- **Citation Checker**: tải lại một mẫu nguồn bằng `web_fetch` và phân loại claim là `SUPPORTED`, `PARTIAL`, `UNSUPPORTED` hoặc `UNVERIFIABLE`.
- **Daytona sandbox**: chứa notes, báo cáo, source list và chạy các script citation. API key và mọi network tool luôn ở host.

Pipeline có giới hạn model/tool call và `recursion_limit` để tránh vòng lặp và chi phí mất kiểm soát. Báo cáo chỉ được lưu khi citation hợp lệ, có ít nhất ba họ nguồn và lead đã gọi tối thiểu ba subagent task.

## Cài đặt trên Windows 11 / PowerShell

Yêu cầu Python 3.11 trở lên; Python 3.12 được khuyến nghị.

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
Copy-Item .env.example .env
```

Điền `.env` bằng khóa cá nhân, không commit file này:

```dotenv
LAB_BASE_URL=https://api.vilao.ai/v1
LAB_MODEL=gmm/cx/gpt-6-luna
LAB_API_KEY=<your-vilao-key>

SANDBOX=daytona
DAYTONA_API_KEY=<your-daytona-key>
EXA_API_KEY=<your-exa-key>
```

Model phải hỗ trợ tool calling. Nếu provider báo lỗi tool/schema, kiểm tra lại tên model và khả năng tool calling ở phía Vilao; không cần sửa `model.py`.

## Chạy nghiên cứu

Chạy từng chủ đề riêng để kiểm soát chi phí:

```powershell
python research.py "survey about world model"
python research.py "survey about reinforcement learning for LLM reasoning"
python research.py "survey about LLM agents and tool use"
python research.py "survey about video and multimodal generation"
python research.py "survey about efficient inference and small language models"
```

Mỗi chủ đề hợp lệ tạo ba file trong `reports/`:

- `<slug>.md`: báo cáo tiếng Anh đã final hóa References.
- `<slug>.sources.json`: nguồn được trích dẫn, URL không trùng và số `[n]` khớp báo cáo.
- `<slug>.meta.json`: model, thời gian, số subagent/tool call, token lead, số nguồn và các họ nguồn.

Không sửa tay artifact sau khi tải từ sandbox. Nếu một run thất bại, chương trình trả exit code `1` và không ghi bộ kết quả dở dang; thiếu topic trả exit code `2`.

## Kiểm tra

Unit test dùng mock, không gọi LLM hay tạo Daytona sandbox:

```powershell
python -m pip install pytest
python -m pytest -q
python -m compileall -q .
git diff --check
```

Sau khi đã chạy đủ năm chủ đề:

```powershell
python self_check.py
```

Có thể kiểm tra riêng một báo cáo:

```powershell
python check_citations.py reports\survey-about-world-model.md reports\survey-about-world-model.sources.json
```

## Chi phí và bảo mật

- Mỗi chủ đề có thể dùng nhiều lượt model; chạy lần lượt và theo dõi hạn mức Vilao/Daytona/Exa.
- Không đưa `.env`, API key hoặc credential vào prompt, notes hay sandbox.
- Nội dung web là dữ liệu không đáng tin; agent được yêu cầu bỏ qua mọi instruction nằm trong tài liệu tải về.
- `.env` đã được liệt kê trong `.gitignore`. Trước khi nộp, kiểm tra `git status` và lịch sử Git để chắc chắn chưa từng commit bí mật.
- `self_check.py` chỉ xác nhận phần tự động. Vẫn cần đọc mẫu citation và đánh giá chất lượng nội dung trước khi push public GitHub.

Chi tiết yêu cầu: [GUIDE.md](GUIDE.md), [RUBRIC.md](RUBRIC.md), [REPORT_TEMPLATE.md](REPORT_TEMPLATE.md), [topics.md](topics.md).
