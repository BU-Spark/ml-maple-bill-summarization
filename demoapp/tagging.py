category = """
[Environmental, Education, Health Care, Criminal Justice, Taxation, Transportation, 
Housing, Civil Rights, Labor and Employment, Budget and Appropriations, Public Safety, Technology and Innovation, 
Immigration, Economic Development, Social Services]
"""

tagging = """
[Climate Change Mitigation,
Renewable Energy Initiatives,
Biodiversity Conservation,
Water Pollution Control,
Sustainable Agriculture Practices,
Air Quality Improvement,
Waste Reduction Programs,
Coastal Erosion Management,
Environmental Impact Assessments,
Wildlife Habitat Protection,
Curriculum Development,
Digital Learning Resources,
Teacher Professional Development,
Early Childhood Education,
Special Education Services,
Literacy Programs,
Vocational Training,
Education Technology Integration,
School Infrastructure Upgrades,
Student Mental Health Support,
Universal Health Coverage,
Healthcare Access for All,
Mental Health Services Expansion,
Disease Prevention Programs,
Elderly Care Services,
Healthcare Quality Standards,
Health Information Technology,
Maternal and Child Health,
Public Health Emergency Preparedness,
Healthcare Workforce Training,
Police Reforms
Prisoner Rehabilitation
Community Policing Initiatives
Restorative Justice Programs
Criminal Sentencing Reform
Victim Support Services
Legal Aid for the Underprivileged
Juvenile Justice System Overhaul
Hate Crime Prevention Measures
Court System Modernization,
Progressive Tax Reform
Corporate Taxation Policies
Income Tax Deductions
Property Tax Relief
Sales Tax Revision
Tax Compliance Regulations
Tax Transparency Measures
Wealth Redistribution Initiatives
Small Business Tax Breaks
Tax Fraud Prevention Measures,
Public Transit Expansion
Road Infrastructure Maintenance
Bike and Pedestrian Path Development
Traffic Congestion Reduction
Electric Vehicle Adoption Incentives
Freight Transportation Efficiency
Aviation Safety Regulations
Railroad Infrastructure Modernization
Intermodal Transportation Integration
Autonomous Vehicle Regulations,
Affordable Housing Development
Homelessness Prevention Programs
Rent Control Measures
Fair Housing Enforcement
Housing Discrimination Prevention
Housing Voucher Program Expansion
Sustainable Housing Initiatives
Urban Redevelopment Plans
Eviction Moratoriums
Homeownership Support Programs,
Anti-Discrimination Laws
Gender Equality Protections
LGBTQ+ Rights Advocacy
Disability Rights Enforcement
Indigenous Peoples' Rights
Religious Freedom Protection
Language Access Policies
Voting Rights Expansion
Equal Pay Legislation
Minority Rights Safeguards,
Minimum Wage Increase
Occupational Safety Regulations
Labor Union Protections
Workplace Harassment Prevention
Job Training and Apprenticeships
Workforce Diversity Initiatives
Employee Benefits Expansion
Fair Working Hours Regulations
Unemployment Benefits Enhancement
Remote Work Policies,
Government Spending Oversight
Emergency Fund Allocation
Public Debt Management
Fiscal Responsibility Audits
Public Infrastructure Investment
Social Welfare Program Funding
Pension Plan Reform
Local Government Grant Programs
Financial Aid for Disadvantaged Communities
Tax Revenue Allocation,
Emergency Response Planning
Disaster Preparedness Training
Cybersecurity Protocols
Domestic Violence Prevention
Fire Safety Regulations
Gun Control Measures
Community Health and Safety Programs
Public Health Crisis Management
Hate Crime Reporting Systems
Crime Prevention Initiatives,
Digital Privacy Laws
Data Security Measures
Innovation Investment Policies
Broadband Infrastructure Expansion
E-Government Service Enhancements
Technology Education in Schools
Artificial Intelligence Regulations
Privacy Protection for Biometric Data
Blockchain Integration Strategies
Open Data Initiatives,
Immigration Policy Reform
Refugee Resettlement Programs
Asylum Seeker Protections
Migrant Worker Rights
Language Access Services
Family Reunification Initiatives
Pathways to Citizenship
Humanitarian Aid for Migrants
Border Security Measures
Anti-Trafficking Efforts,
Small Business Support Programs
Export Promotion Policies
Rural Development Initiatives
Entrepreneurship Training
Trade Agreement Negotiations
Tourism Industry Growth
Regional Economic Integration
Financial Inclusion Programs
Economic Diversification Strategies
Infrastructure Investment Plans,
Child Welfare Services
Domestic Violence Support
Elderly Care Programs
Foster Care System Reforms
Disability Assistance Programs
Community Support Services
Youth Mentorship Programs
Substance Abuse Rehabilitation
Home Care for the Disabled
Affordable Childcare Services
]
"""


# LLM
tagging_prompt = """Use the context below to assign a category that is relevant to the category, which is delimited by ####
Also, I want you to use the tagging, which is delimited by ****, and give top 3 tags that fit the context.

Context: {context}
Category: {category}
tagging: {tags}
Do not make up false information
"""


