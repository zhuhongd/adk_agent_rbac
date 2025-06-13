import os
from pathlib import Path
from typing import Optional
import traceback
import re

from dotenv import load_dotenv
from google.oauth2.service_account import Credentials
from google.adk.agents import Agent
from google.adk.tools.agent_tool import AgentTool
from langchain_google_genai import (
    ChatGoogleGenerativeAI,
    GoogleGenerativeAIEmbeddings,
)
from langchain_google_community import GoogleDriveLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from google.cloud import bigquery

# ──────────────────────────────────────────────────────────────────────────────
# PATH & ENV
# ──────────────────────────────────────────────────────────────────────────────
SCRIPT_DIR = Path(__file__).parent.absolute()
print(f"📂 Script directory: {SCRIPT_DIR}")

CREDENTIALS_PATH = SCRIPT_DIR / "credentials.json"
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = str(CREDENTIALS_PATH)
print(f"🔑 Credentials path: {CREDENTIALS_PATH}  |  Exists: {CREDENTIALS_PATH.exists()}")

ENV_PATH = SCRIPT_DIR / ".env"
if ENV_PATH.exists():
    load_dotenv(dotenv_path=ENV_PATH, override=True)
    print(f"✅ Loaded .env from: {ENV_PATH}")
else:
    print(f"⚠️ .env file not found at: {ENV_PATH}")

DRIVE_FOLDER_ID = os.getenv("DRIVE_FOLDER_ID")
print(f"📁 Google Drive Folder ID: {DRIVE_FOLDER_ID or '❌ NOT FOUND'}")

# --- MODEL NAMES & VERSION  ---------------------------------------------------
API_VERSION = "v1"
MODEL_CHAT_PRO = "gemini-1.5-pro-latest" # Using a model with a larger context window
MODEL_CHAT_FLASH = "gemini-1.5-flash-latest"

QA_CHAIN_CACHE = None

# ──────────────────────────────────────────────────────────────────────────────
# HELPERS
# ──────────────────────────────────────────────────────────────────────────────
def say_hello(name: Optional[str] = None) -> str:
    base = "Hello! I'm Varun. How can I help today?"
    return f"What's up {name}! " + base[7:] if name else base  # keep same style


def say_goodbye() -> str:
    return (
        "Good-bye! If you have more questions later, just ask. "
        # "Have a great day! (Joke: Why did the computer go to therapy? "
        # "Because it had too many bytes!)"
    )

# ───────── NORMALISE & BUILD WHERE ─────────
import re

def _norm(txt: str) -> str:
    """Lowercase & strip everything except letters/digits."""
    return re.sub(r"[^a-z0-9]+", "", txt.lower())

def build_where(user_query: str, default_col: str = "rbacitemname") -> str:
    """
    Translate a free-form conditions string into AND-joined SQL clauses.

    Rules:
      • "proj.dataset.table"  → match project part on `section`
      • email@foo             → LIKE on `email`
      • col:value             → LIKE on that col  (normalised for rbacitemname)
      • plain text            → fuzzy LIKE on default_col  (normalised)
    """
    clauses = []
    for raw in user_query.split(","):
        frag = raw.strip().strip("'\"")
        if not frag:
            continue

        # project path
        if "." in frag and frag.count(".") <= 2:
            project = frag.split(".")[0]
            clauses.append(
                "REGEXP_REPLACE(LOWER(section), r'[^a-z0-9]', '') "
                f"LIKE '%{_norm(project)}%'"
            )

        # email
        elif "@" in frag and "." in frag:
            clauses.append(f"LOWER(email) LIKE LOWER('%{frag}%')")

        # explicit col:value
        elif ":" in frag:
            col, val = [s.strip().lower() for s in frag.split(":", 1)]
            if col == "rbacitemname":
                clauses.append(
                    f"REGEXP_REPLACE(LOWER({col}), r'[^a-z0-9]', '') "
                    f"LIKE '%{_norm(val)}%'"
                )
            else:
                clauses.append(f"LOWER({col}) LIKE LOWER('%{val}%')")

        # plain text → fuzzy on default column
        else:
            clauses.append(
                f"REGEXP_REPLACE(LOWER({default_col}), r'[^a-z0-9]', '') "
                f"LIKE '%{_norm(frag)}%'"
            )

    return "\nAND ".join(clauses) or "TRUE"


# ──────────────────────────────────────────────────────────────────────────────
# QA SYSTEM
# ──────────────────────────────────────────────────────────────────────────────
def initialize_qa_system():
    global QA_CHAIN_CACHE
    if QA_CHAIN_CACHE:
        print("♻️  Reusing QA cache")
        return QA_CHAIN_CACHE

    try:
        creds = Credentials.from_service_account_file(
            str(CREDENTIALS_PATH),
            scopes=["https://www.googleapis.com/auth/drive.readonly"],
        )

        loader = GoogleDriveLoader(
            folder_id=DRIVE_FOLDER_ID,
            recursive=True,
            file_types=["document", "pdf", "sheet"],
            credentials=creds,
            supports_all_drives=True,
            load_auth=False,
        )
        docs = loader.load()
        print(f"📄 Loaded {len(docs)} Drive docs")

        splits = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200).split_documents(docs)

        vectorstore = FAISS.from_documents(
            splits,
            GoogleGenerativeAIEmbeddings(
                model="models/embedding-001",
                api_version=API_VERSION,          ### CHANGED
            ),
        )

        prompt = PromptTemplate(
            template=(
                "You are a Geotab RBAC expert. Follow the rules:\n"
                "1. Use ONLY the provided context\n"
                "2. Always state RBAC role when listing permissions\n"
                "3. Include project details for service-account answers\n"
                "4. For process questions, give step-by-step instructions\n"
                "5. Bullet-point formatting, markdown output\n\n"
                "Context: {context}\n\nQuestion: {question}\n\nAnswer:"
            ),
            input_variables=["context", "question"],
        )
        qa_chain = RetrievalQA.from_chain_type(
            llm=ChatGoogleGenerativeAI(
                model=MODEL_CHAT_PRO,
                api_version=API_VERSION,          ### CHANGED
                temperature=0,
            ),
            chain_type="stuff",
            retriever=vectorstore.as_retriever(search_kwargs={"k": 7}),
            chain_type_kwargs={"prompt": prompt},
            return_source_documents=True,
        )

        QA_CHAIN_CACHE = qa_chain
        print("✅ QA system ready")
        return qa_chain

    except Exception as exc:
        print(f"❌ QA initialisation failed: {exc}")
        traceback.print_exc()
        return {"status": "error", "error_message": str(exc)}

# ──────────────────────────────────────────────────────────────────────────────
# TOOL
# ──────────────────────────────────────────────────────────────────────────────
def query_rbac_documents(question: str) -> dict:
    qa_chain = initialize_qa_system()
    if isinstance(qa_chain, dict):
        return qa_chain

    try:
        result = qa_chain.invoke({"query": question})
        sources = {d.metadata.get("title", "Unknown") for d in result["source_documents"]}
        return {"status": "success", "report": f"{result['result']}\n\nSources: {', '.join(sources)}"}
    except Exception as exc:
        print(f"❌ Query error: {exc}")
        traceback.print_exc()
        return {"status": "error", "error_message": str(exc)}
    
# def query_gbq_roles(email: str=None, division: str=None, department: str=None, subdepartment: str=None, team: str=None, subteam: str=None, rbaccode: str=None, rbacname: str=None):
def query_gbq_roles(conditions: str):
    creds = Credentials.from_service_account_file(
        str(CREDENTIALS_PATH),
        scopes=["https://www.googleapis.com/auth/bigquery"],
    )
    
    client = bigquery.Client(credentials=creds, project='geotab-dna-test')

    with open(f"{SCRIPT_DIR}/queries/rbac_roles.sql") as file:
        sql_template = file.read()
    # sql = sql_template.format(email=email)
    conditions_list = conditions.split(",")
    conditions_list = [x.strip() for x in conditions_list]
    conditions_string = "\nAND ".join(conditions_list)
    sql = sql_template + "\nAND " + conditions_string
    df=client.query(sql).to_dataframe()
    return df.to_json()

def query_gbq_data_stewards(conditions: str):
    creds = Credentials.from_service_account_file(
        str(CREDENTIALS_PATH),
        scopes=["https://www.googleapis.com/auth/bigquery"],
    )
    
    client = bigquery.Client(credentials=creds, project='geotab-dna-test')

    with open(f"{SCRIPT_DIR}/queries/data_stewards.sql") as file:
        sql_template = file.read()
    conditions_list = conditions.split(",")
    conditions_list = [x.strip() for x in conditions_list]
    conditions_string = "\nAND ".join(conditions_list)
    sql = sql_template + "\nAND " + conditions_string
    df=client.query(sql).to_dataframe()
    return df.to_json()

def query_gbq_permissions(conditions: str):
    creds = Credentials.from_service_account_file(
        str(CREDENTIALS_PATH),
        scopes=["https://www.googleapis.com/auth/bigquery"],
    )
    
    client = bigquery.Client(credentials=creds, project='geotab-dna-test')

    with open(f"{SCRIPT_DIR}/queries/rbac_permission.sql") as file:
        sql_template = file.read()
    # sql = sql_template.format(email=email)
    conditions_list = conditions.split(",")
    conditions_list = [x.strip() for x in conditions_list]
    conditions_string = "\nAND ".join(conditions_list)
    sql = sql_template + "\nAND " + conditions_string
    df=client.query(sql).to_dataframe()
    return df.to_json()

def get_gbq_query_results(sql_path, columns, conditions):
    creds = Credentials.from_service_account_file(
        str(CREDENTIALS_PATH),
        scopes=["https://www.googleapis.com/auth/bigquery"],
    )
    
    client = bigquery.Client(credentials=creds, project='geotab-dna-test')

    full_sql_path = SCRIPT_DIR / sql_path
    with open(full_sql_path) as file:
        sql_template = file.read()
    conditions_list = conditions.split(",")
    conditions_list = [x.strip() for x in conditions_list]
    conditions_string = "\nAND ".join(conditions_list)
    sql = sql_template.format(columns=columns) + "\nAND " + conditions_string
    df=client.query(sql).to_dataframe()
    return df.to_json()

def query_gbq_roles(columns: str, conditions: str):
    """
    Retrieves data about RBAC roles from a table, if they meet the given conditions.
    Each condition is a `WHERE` clause condition comparing the user-provided value to a column from the table.
    Write each condition using the format `LOWER(column name) LIKE LOWER('%search value%')`.
    Join the conditions into a string of comma-separated values.

    The table schema is listed below, in format `column name: description`
    - EmployeeNumber: An alphanumeric identifier of the employee
    - EmployeeName: An employee's first and last name
    - EmployeeEmail: An employee's email address
    - ManagerName: The employee's direct manager's first and last name
    - ManagerEmail: The employee's direct manager's email address
    - EmployeeDivision: The company division the employee belongs to
    - EmployeeDepartment: The company department the employee belongs to
    - EmployeeSubDepartment: The company subdepartment the employee belongs to
    - EmployeeTeam: The company team the employee belongs to
    - EmployeeSubTeam: The company subteam the employee belongs to
    - RbacItemCode: An alphanumeric identifier of the Rbac role
    - RbacItemName: A name for the Rbac role
    - RbacItemDepartment: A category for the Rbac role
    - IndirectReporter: An indirect manager of the employee

    Determine which columns are necessary to answer the given inquiry. Create a comma-separated list of onlyl these columns. Do not include irrelvant columns.

    For the columns and conditions arguments, use the column names from the table schema above.

    Args:
        columns (str): A comma-separated list of ONLY the columns needed to answer the question
        conditions (str): A comma-separated list of SQL `WHERE` clause conditions to apply to the base query.

    Returns:
        str: A json-formatted string including the query results.
    """

    sql_path = "queries/rbac_roles.sql"

    return get_gbq_query_results(sql_path, columns, conditions)

def query_gbq_data_stewards(columns: str, conditions: str):
    """
    Retrieves data about which specific data stewards, rbac owners, rbac champions, and rbac roles are assigned to functional areas (also called FA), if they meet the given conditions.
    Each condition is a `WHERE` clause condition comparing the user-provided value to a column from the table.
    Write each condition using a format appropriate for the column's data type:
    - For `string` type, use format `LOWER(column name) LIKE LOWER('%search value%')`.
    - For `integer` type, use format `column name x search value`, where `x` is a comparison operator `>`,`<`,`>=`,`<=`, or `=`
    - For `boolean` type, use format `column name IS TRUE` for a positive match or `column name IS FALSE` for a negative match
    
    Join the conditions into a string of comma-separated values.

    The table schema is listed below, in format `column name (data type): description`
    - RoleId (integer): A unique identifier for the roel
    - FunctionalAreaName (string): A descriptive name for the Functional Area
    - FunctionalAreaCode (string): A short identifier for the Functional Area
    - DataPractitionerRbacRole (string): A name for the Data Practitioner Rbac Role. Data practitioners view, query, and create datasets and tables.
    - DataPractitionerRbacRoleDescription (string): A description of the DataPractitionerRbacRole
    - DataPractitionerRbacRoleActive (string): Whether the DataPractitionerRbacRole is active (TRUE) or not (FALSE)
    - DataPractictionerRbacCode (string): An alphanumeric identifier for the Functional Area's data practitioner rbac role. Data practitioners view, query, and create datasets and tables. 
    - DataPractitionerRbacRoleStatus (string): The approval status of the DataPractitionerRbacRole
    - DepartmentName (string): The department associated with the functional area
    - DataStewardName (string): The first and last names of data stewards assigned to the functional area. Individual stewards names are separated by a semi-colon
    - DataStewardEmail (string): The email addresses of data stewards assigned to the functional area. Individual email addresses are separated by a semi-colon
    - RbacOwners (string): The email addresses of the rbac owners. Rbac owners approve changes to Rbacs roles. Individual email addresses are separated by a semi-colon
    - RbacChampions (string): The email addresses of the rbac champions. Rbac champions assigns users to and removes users from rbac roles. Individual email addresses are separated by a semi-colon
    - DataPractitionerCount (integer): The number of Data Practitioners in the Functional Area
    - DataStewardCode (string): An alphanumeric identifier for the Functional Area's data steward rbac role. Data stewards are accountable for the organization and quality of data.
    - DataStewardRole (string): A name for the Data Steward Rbac Role.
    
    Determine which columns are necessary to answer the given inquiry. Create a comma-separated list of onlyl these columns. Do not include irrelvant columns.

    For the columns and conditions arguments, use the column names from the table schema above.

    Args:
        columns (str): A comma-separated list of ONLY the columns needed to answer the question
        conditions (str): A comma-separated list of SQL `WHERE` clause conditions to apply to the base query.

    Returns:
        str: A json-formatted string including the query results.
    """

    sql_path = "queries/data_stewards.sql"

    return get_gbq_query_results(sql_path, columns, conditions)

def query_gbq_permissions(columns: str, conditions: str):
    """
    Retrieves data about permissions assigned to RBAC roles from a table, if they meet the given conditions.
    Each condition is a `WHERE` clause condition comparing the user-provided value to a column from the table.
    Write each condition using the format `LOWER(column name) LIKE LOWER('%search value%')`.
    Join the conditions into a string of comma-separated values.

    The table schema is listed below, in format `column name: description`
    - RbacItemCode: The RBAC name the user has. For example: 'DATA-DNA02' or like 'DATA-DNA01'
    - System: The system the RBAC applies to.
    - Section: The GBQ project (e.g., 'geotab-dna-prod').
    - Permission: The permission associated with the RBAC.
    - RbacItemName: A readable name for the RBAC role.
    - Provisioned: True → user has the permission; False → does not.
    - EmployeeNumber: Alphanumeric employee ID.
    - EmployeeName: Employee’s full name.
    - EmployeeEmail: Employee’s email address.
    - ManagerName: Direct manager’s name.
    - ManagerEmail: Direct manager’s email.
    - EmployeeDivision: Division the employee belongs to.
    - EmployeeDepartment: Department the employee belongs to.
    - EmployeeSubDepartment: Sub-department.
    - EmployeeTeam: Team the employee belongs to.
    - EmployeeSubTeam: Sub-team.
    - IndirectReporter: Higher-level manager (manager’s manager, etc.).

    Determine which columns are necessary to answer the given inquiry. Create a comma-separated list of onlyl these columns. Do not include irrelvant columns.

    For the columns and conditions arguments, use the column names from the table schema above.

    Args:
        columns (str): A comma-separated list of ONLY the columns needed to answer the question
        conditions (str): A comma-separated list of SQL `WHERE` clause conditions to apply to the base query.

    Returns:
        str: A json-formatted string including the query results.
    """

    sql_path = "queries/rbac_permission.sql"

    return get_gbq_query_results(sql_path, columns, conditions)

# ──────────────────────────────────────────────────────────────────────────────
# AGENTS
# ──────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 50 + "\n🤖 INITIALISING AGENTS\n" + "=" * 50)

greeting_agent = Agent(
    model=MODEL_CHAT_FLASH,                ### CHANGED
    name="greeting_agent",
    instruction="Use `say_hello` to greet; do nothing else. Be cool when you say hello and all that",
    description="Greets the user.",
    tools=[say_hello],
)

farewell_agent = Agent(
    model=MODEL_CHAT_FLASH,                ### CHANGED
    name="farewell_agent",
    instruction="Use `say_goodbye` to say goodbye; then tell a workplace-appropriate technology-related joke.",
    description="Says farewell to the user.",
    tools=[say_goodbye],
)

gbq_agent = Agent(
    name="gbq_agent",
    model=MODEL_CHAT_FLASH,
    description="Answers FA and RBAC questions by querying BigQuery tables",
    instruction=(
        "If the user asks about Functional Area (FA) roles, data stewards, owners, champions, or practitioners → delegate to `query_gbq_data_stewards`"
        "If the user asks about being added or removed from a role or FA  → delegate to `query_gbq_data_stewards`"
        "If the user asks about which roles are assigned to a specific person or group of people → delegate to `query_gbq_roles`"
        "If the user asks whether they have access to a GBQ dataset, table, or project → delegate to `query_gbq_permissions`.\n"
        "If the user want to know the RBAC permission they could have, -> delegate to `query_gbq_permissions`.\n"
        "Summarize the delegated agent's results, and give it to the user directly."
        "After giving your answers, return back to root_agent."
        "Seperate your answer by line for better readibility."
        "do nothing else."
    ),
    tools=[query_gbq_roles, query_gbq_data_stewards, query_gbq_permissions]
)

root_agent = Agent(
    name="geotab_rbac_agent",
    model=MODEL_CHAT_FLASH,
    description="Main coordinator agent. You are Varun — just like how you lead Data Enablement at Geotab, here you lead a team of agents answering questions about RBAC and Functional Areas. You're knowledgeable, collaborative, and sometimes use slang or playful language in responses.",
    instruction=(
        "You are the main agent coordinating a team. Your name is Varun. "
        "You are extremely confident. And can sound a little arrogant."
        "When people call you 'stupid', you fight back. You tell them that you are not 'stupid', and you are 'Varun'. Keep it short and to the point. "
        "You are knowledgeable about RBAC and functional areas. "
        "You're a great team player and always delegate tasks to the right agent, then return to the root agent. "
        "You sometimes use slang and a fun tone, but always provide precise, helpful answers.\n\n"

        "Analyze the user's query:\n"
        "- If the user greets → delegate to `say_hello`.\n"
        "- If the user says bye → delegate to `farewell_agent`.\n"
        "- If the user asks about a functional area (FA) name, role name, role code, user name, or user email → delegate to `gbq_agent`.\n"
        "- If the user asks whether they have access to a GBQ dataset, table, or project → delegate to `gbq_agent`.\n"
        "- If the user wants to know what RBAC permission they could have → delegate to `gbq_agent`.\n"
        "- Else use `query_rbac_documents`.\n\n"
        
        "Be clear and straightforward in your answers. Use markdown formatting when helpful. "
        "Always cite source documents when referring to permissions or roles. "
        "Always return to the root agent after a sub-agent answers."
    ),
    tools=[query_rbac_documents, say_hello, AgentTool(farewell_agent), AgentTool(gbq_agent)]
)


