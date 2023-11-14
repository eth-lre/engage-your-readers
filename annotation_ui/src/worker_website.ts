import { log_data, get_exit_questions } from "./connector";
import { DEVMODE } from "./globals";
import { range, timer } from "./utils";

let main_text_area = $("#main_text_area")
let progress_screens = 0;

function check_unlocks() {
    let unlock_next = Object.keys(globalThis.responses).every((k, index, array) => {
        return Object.keys(globalThis.responses[k]).length >= 6
    })
    if (unlock_next) {
        $("#but_next").prop("disabled", false);
    }
}

async function setup_main_text() {
    let frame_obj = $(`<div id="article_frame">${globalThis.data_now["article"]}</div>`)
    let question_obj = $('<div id="main_question_panel"></div>')
    main_text_area.html("")
    main_text_area.append(frame_obj)
    main_text_area.append(question_obj)
    main_text_area.scrollTop(0)
    
    // hack for JS event loop
    await timer(10)

    globalThis.data_now["questions_intext"].forEach(async (element, element_i) => {
        let question_span = $(`#question_${element_i}`)
        question_span.append("&nbsp;")
        // hack for JS event loop
        await timer(10)

        let offset_x = question_span.position().left
        let offset_y = question_span.position().top
        
        question_obj.append(`
            <hr class="arrow" style="width: 0px; height: 8px; position: absolute; left: ${offset_x+1}px; top: ${offset_y-8}px;">
            <hr class="line" style="width: ${1000-offset_x}px; position: absolute; left: ${offset_x+1}px; top: ${offset_y-10}px;">
            <div class="question_box" style="position: absolute; top: ${offset_y-3}px">
                ${element["question"]}
            </div>
        `)
    })
    // <br>
    // <input type="text">


    $("#but_next").on("click", () => {
        progress_screens += 1;
        switch (progress_screens) {
            case 1: setup_performance_questions(); break;
            case 2: setup_exit_questions(); break;
            case 3: load_thankyou(); $("#phases_area").remove(); break;
        }
    })
}

async function setup_performance_questions() {
    $("#instruction_area").html("Please answer the following questions.")
    $("#progress_area").html("Answer all questions before continuing.")
    main_text_area.scrollTop(0)
    main_text_area.html("")

    let questions = globalThis.data_now["questions_performance"]
    questions.forEach((question) => {
        main_text_area.append(`
            <div class="performance_question_text">${question["question"]}</div>
            
        `)
        main_text_area.append(`
            <textarea class='performance_question_value' placeholder='Please provide a detailed answer'></textarea>
            <br><br>
        `)
    })
}

async function setup_exit_questions() {
    main_text_area.scrollTop(0)
    main_text_area.html("")

    let questions = await get_exit_questions()

    questions.forEach((question) => {
        main_text_area.append(`
            <div class="performance_question_text">${question["question"]}</div>
            
        `)
        if (question["type"] == "text") {
            // add checkboxes if exists
            if (question["checkboxes"].length != 0) {
                main_text_area.append(`<br>`)
            }
            question["checkboxes"].forEach((checkbox, checkbox_i) => {
                main_text_area.append(`
                    <input type="checkbox" id="q_${question["id"]}_${checkbox_i}" name="q_${question["id"]}_${checkbox_i}" value="${checkbox}">
                    <label for="q_${question["id"]}_${checkbox_i}">${checkbox}</label>
                `)
            })
            if (question["checkboxes"].length != 0) {
                main_text_area.append(`<br>`)
            }

            main_text_area.append(`
                <textarea class='performance_question_value' placeholder='Please provide a detailed answer'></textarea>
            `)
        } else if (question["type"] == "likert") {
            let joined_labels = range(1, 7).map((x) => `<label for="likert_${question["id"]}_${x}" value="${x}">${x}</label>`).join("\n")
            let joined_inputs = range(1, 7).map((x) => `<input type="radio" name="likert_${question["id"]}" id="likert_${question["id"]}_${x}" value="${x}" />`).join("\n")

            main_text_area.append(`
                <div class='performance_question_likert_parent'>
                    <div class="performance_question_likert_labels">${joined_labels}</div>
            
                    <span class="performance_question_likert_label" style="text-align: right">${question["labels"][0]}</span>
                    ${joined_inputs}
                    <span class="performance_question_likert_label" style="text-align: left">${question["labels"][1]}</span>
                </div>
            `)
        } else if (question["type"] == "intext_questions") {
            globalThis.data_now["questions_intext"].forEach((question, question_i) => {
                main_text_area.append(`
                    <input type="checkbox" id="q_${question["id"]}_${question_i}" value="${question_i}">
                    <label class="exit_questions_intext" for="q_${question["id"]}_${question_i}">${question["question"]}</label>
                    <br>
                `)
            })
            main_text_area.append(`
                <br>
                <textarea class='performance_question_value' placeholder='Please provide a detailed answer'></textarea>
            `)
        }
        main_text_area.append("<br><br>")
    })
}

function load_cur_text() {
    globalThis.responses = {}
    globalThis.responsesExpectedCountUnlock = 0
    globalThis.responsesExpectedCount = 0
    globalThis.data_now["start_time"] = Date.now()
    setup_main_text()
}

async function load_thankyou() {
    $("#instruction_area").hide()

    main_text_area.html("Please wait 3s for data synchronization to finish.")
    await timer(1000)
    main_text_area.html("Please wait 2s for data synchronization to finish.")
    await timer(1000)
    main_text_area.html("Please wait 1s for data synchronization to finish.")
    await timer(1000)

    let html_text = `Thank you for participating in our study. Please get back to the experiment manager.`;
    if (globalThis.uid.startsWith("prolific_pilot_1")) {
        html_text += `<br>Please click <a href="https://app.prolific.co/submissions/complete?cc=C1FV7L5F">this link</a> to go back to Prolific. `
        html_text += `Alternatively use this code <em>C1FV7L5F</em>.`
    }
    main_text_area.html(html_text);
}

export { load_cur_text }