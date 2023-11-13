import { log_data } from "./connector";
import { DEVMODE } from "./globals";
import { timer } from "./utils";

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
    await timer(100)
    // frame_obj.contents().find('html').html(globalThis.data_now["article"]);

    let offset_x_main = main_text_area.position().left
    let offset_y_main = main_text_area.position().top

    let current_offset_y = 0;
    globalThis.data_now["questions_intext"].forEach((element, element_i) => {
        let offset_x = $(`#question_${element_i}`).position().left
        let offset_y = $(`#question_${element_i}`).position().top
        console.log(offset_x-50)
        
        question_obj.append(`
            <hr class="arrow" style="width: 1px; position: absolute; left: ${offset_x}px; top: ${offset_y+8}px;">
            <hr class="line" style="width: ${1000-offset_x}px; position: absolute; left: ${offset_x}px; top: ${offset_y+10}px;">
            <div class="question_box" style="position: absolute; top: ${offset_y}px">
                ${element["question"]}
            </div>
        `)
    })
    // <br>
    // <input type="text">


    $("#but_next").on("click", () => {
        console.log("clicking")
        progress_screens += 1;
        switch (progress_screens) {
            case 1: setup_performance_questions(); break;
            case 2: setup_exit_questions(); break;
            case 3: load_thankyou(); $("#phases_area").remove(); break;
        }
    })
}

async function setup_performance_questions() {
    $("#instruction_area").html("Please answer the following questions")
    main_text_area.html("Performance questions")
}

async function setup_exit_questions() {
    main_text_area.html("")

    let result = await $.ajax(
        "exit_questions.jsonl",
        {
            type: 'GET',
            contentType: 'application/text',
        }
    )
    result = result.trimEnd()
    result = "[" + result.replaceAll("\n", ",") + "]"
    result = JSON.parse(result)

    result.forEach((question_data) => {
        main_text_area.append(`
            <div class="performance_question_text">${question_data["question"]}</div>
            
        `)
        if (question_data["type"] == "text") {
            // add checkboxes if exists
            if (question_data["checkboxes"].length != 0) {
                main_text_area.append(`<br>`)
            }
            question_data["checkboxes"].forEach((checkbox, checkbox_i) => {
                main_text_area.append(`
                <input type="checkbox" id="q_${question_data["id"]}_${checkbox_i}" name="q_${question_data["id"]}_${checkbox_i}" value="${checkbox}">
                <label for="q_${question_data["id"]}_${checkbox_i}">${checkbox}</label>
                `)
            })
            if (question_data["checkboxes"].length != 0) {
                main_text_area.append(`<br>`)
            }

            main_text_area.append(`
                <textarea class='performance_question_value' placeholder='Please provide a detailed answer'></textarea>
            `)
        } else if (question_data["type"] == "likert") {
            main_text_area.append(`
                <div class='performance_question_likert_parent'>
                    <div class="performance_question_likert_labels">1 2 3 4 5 6 7</div>
            
                    <span class="performance_question_likert_label" style="text-align: right">${question_data["labels"][0]}</span>
                    <input type="radio" name="likert_${question_data["id"]}" value="1" />            
                    <input type="radio" name="likert_${question_data["id"]}" value="2" />
                    <input type="radio" name="likert_${question_data["id"]}" value="3" />
                    <input type="radio" name="likert_${question_data["id"]}" value="4" />
                    <input type="radio" name="likert_${question_data["id"]}" value="5" />
                    <input type="radio" name="likert_${question_data["id"]}" value="6" />
                    <input type="radio" name="likert_${question_data["id"]}" value="7" />
                    <span class="performance_question_likert_label" style="text-align: left">${question_data["labels"][1]}</span>
                </div>
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

function load_thankyou() {
    // TODO: wait for data sync
    let html_text = `Thank you for participating in our study. `;
    if (globalThis.uid.startsWith("prolific_pilot_1")) {
        html_text += `<br>Please click <a href="https://app.prolific.co/submissions/complete?cc=C1FV7L5F">this link</a> to go back to Prolific. `
        html_text += `Alternatively use this code <em>C1FV7L5F</em>.`
    }
    main_text_area.html(html_text);
}

export { load_cur_text }