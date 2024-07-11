from ibm_watson_machine_learning.foundation_models import Model
import gradio as gr


model_id = "meta-llama/llama-2-70b-chat"

my_credentials = {
    "url": "https://us-south.ml.cloud.ibm.com"
}

gen_parms = {
    "max_new_tokens": 512,  
    "temperature": 0.7  
}
project_id = "skills-network" 
space_id = None
verify = False

model = Model(model_id, my_credentials, gen_parms, project_id, space_id, verify)

def polish_resume(position_name, resume_content, polish_prompt=""):
    if polish_prompt and polish_prompt.strip():
        prompt_use = f"Given the resume content: '{resume_content}', polish it based on the following instructions: {polish_prompt} for the {position_name} position."
    else:
        prompt_use = f"Suggest improvements for the following resume content: '{resume_content}' to better align with the requirements and expectations of a {position_name} position. Return the polished version, highlighting necessary adjustments for clarity, relevance, and impact in relation to the targeted role."
    
    generated_response = model.generate(prompt_use)
    generated_text = generated_response["results"][0]["generated_text"]
    return generated_text

resume_polish_application = gr.Interface(
    fn=polish_resume,
    allow_flagging="never", # Deactivate the flag function in gradio as it is not needed.
    inputs=[
        gr.Textbox(label="Position Name", placeholder="Enter the name of the position..."),
        gr.Textbox(label="Resume Content", placeholder="Paste your resume content here...", lines=20),
        gr.Textbox(label="Polish Instruction (Optional)", placeholder="Enter specific instructions or areas for improvement (optional)...", lines=2),
    ],
    outputs=gr.Textbox(label="Polished Content"),
    title="Resume Polish Application",
    description="This application helps you polish your resume. Enter the position your want to apply, your resume content, and specific instructions or areas for improvement (optional), then get a polished version of your content."
)

resume_polish_application.launch()
