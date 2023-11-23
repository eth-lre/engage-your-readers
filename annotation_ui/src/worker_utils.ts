
import { range } from "./utils";

export function instantiate_question(question: object) {
    let output = `<div class="performance_question_text">${question["question"]}</div>`

    // add stub checkboxes if they don't exist
    if (!("checkboxes" in question)) {
        question["checkboxes"] = []
    }
    if (!("checkbox_type" in question)) {
        question["checkbox_type"] = "checkbox"
    }
    if (question["type"].startsWith("text")) {
        if (question["checkboxes"].length != 0) {
            output += (`<br>`)
        }
        question["checkboxes"].forEach((checkbox, checkbox_i) => {
            output += (`
                <input type="${question['checkbox_type']}" id="q_${question["id"]}_${checkbox_i}" name="q_${question["id"]}" value="${checkbox}">
                <label for="q_${question["id"]}_${checkbox_i}">${checkbox}</label>
            `)
        })
        if (question["checkboxes"].length != 0) {
            output += (`<br>`)
        }

        if (question["type"] == "text") {
            output += `<textarea class='performance_question_value' placeholder='Please provide a detailed answer'></textarea>`
        } else if (question["type"] == "text_small") {
            output += `<textarea class='performance_question_small_value'></textarea>`
        }
    } else if (question["type"] == "number") {
        output += `<input type="number" min="15" max="100">`
    } else if (question["type"] == "choices") {
        let options = question["choices"].map((choice) => `<option value="${choice}">${choice}</option>`)
        output += `<select>\n<option value="blank"></option>\n${options.join("\n")}\n</select>`
    } else if (question["type"] == "likert") {
        let joined_labels = range(1, 7).map((x) => `<label for="likert_${question["id"]}_${x}" value="${x}">${x}</label>`).join("\n")
        let joined_inputs = range(1, 7).map((x) => `<input type="radio" name="likert_${question["id"]}" id="likert_${question["id"]}_${x}" value="${x}" />`).join("\n")

        output += (`
            <div class='performance_question_likert_parent'>
                <div class="performance_question_likert_labels">${joined_labels}</div>
        
                <span class="performance_question_likert_label" style="text-align: right">${question["labels"][0]}</span>
                ${joined_inputs}
                <span class="performance_question_likert_label" style="text-align: left">${question["labels"][1]}</span>
            </div>
        `)
    } else if (question["type"] == "intext_questions") {
        globalThis.data_now["questions_intext"].forEach((question, question_i) => {
            output += (`
                <input type="checkbox" id="q_${question["id"]}_${question_i}" value="${question_i}">
                <label class="exit_questions_intext" for="q_${question["id"]}_${question_i}">${question["question"]}</label>
                <br>
            `)
        })
        output += (`
            <br>
            <textarea class='performance_question_value' placeholder='Please provide a detailed answer'></textarea>
        `)
    }
    return output
}
