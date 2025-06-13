# agent_building

## DEMO video:
https://drive.google.com/file/d/1-UdxMt8Vn7c83zvRlIgfwhEgIFtdvbLe/view

## Description
This RBAC agent answers questions about RBAC policy, procedures, and implementation, by referencing documentation and BigQuery resources.

## Usage
Example questions:

- what are the steps to add new person to RBAC?
- When was RBAC SOP last updated
- who can add a user to embedded FA?
- What rbac roles does <user's name or email> have?
- What members of the Data Enablement team have the Revenue Generation role?
- im running into permission errors when i query "geotab-hr-prod", should my RBAC has access to it?

## Support
Reach out to Hongda Zhu (hongdazhu@geotab.com) and Megan Washington (meganwashington@geotab.com)

## Roadmap
- Use logged-in user's identify to answer relative questions (ex. "What are **my** RBAC roles?")
- Add additional data sources, such as dataset clearance levels

## Contributing
State if you are open to contributions and what your requirements are for accepting them.

For people who want to make changes to your project, it's helpful to have some documentation on how to get started. Perhaps there is a script that they should run or some environment variables that they need to set. Make these steps explicit. These instructions could also be useful to your future self.

You can also document commands to lint the code or run tests. These steps help to ensure high code quality and reduce the likelihood that the changes inadvertently break something. Having instructions for running tests is especially helpful if it requires external setup, such as starting a Selenium server for testing in a browser.

## Authors and acknowledgment
Made with 💙 and ☕ by Hongda Zhu and Megan Washington, with support from our manager Varun Naroia

Special thank you to Ian Bigford, Abdulla Hammad, and Gurleen Kaur for providing valuable resources for bringing this project to life.