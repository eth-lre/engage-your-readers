import { log_data } from "./connector";
import { DEVMODE } from "./globals";
import { getIndicies } from "./utils";

let main_text_area = $("#main_text_area")
let progress_screens = 0;

function check_unlocks() {
    let unlock_next = Object.keys(globalThis.responses).every((k, index, array) =>{
        return Object.keys(globalThis.responses[k]).length >= 6
    })
    if (unlock_next) {
        $("#but_next").prop("disabled", false);
    }
}

function setup_main_text() {
    let frame_obj = $('<iframe frameborder="0" scrolling="no" onload="resizeIframe(this)">')
    let question_obj = $('<div id="main_question_panel"></div>')
    main_text_area.html("")
    main_text_area.append(frame_obj)
    main_text_area.append(question_obj)

    // hack for JS event loop
    setTimeout(() => {
        frame_obj.contents().find('html').html(globalThis.data_now["article"]);

        let current_offset = 0;
        globalThis.data_now["questions_intext"].forEach(element => {
            current_offset = element["pos"]-current_offset;
            question_obj.append(`
                <div class="question_box" style="margin-top: ${current_offset}px">
                    ${element["question"]}
                    <br>
                    <input type="text">
                </div>`
            )
        });
    }, 1 );


    $("#but_next").on("click", () => {
        console.log("clicking")
        progress_screens += 1;
        switch(progress_screens) {
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

    result.forEach( (element) => {
        main_text_area.append(`
            <div class="performance_question_text">${element["question"]}</div>
            
        `)
        if (element["type"] == "text") {
            main_text_area.append(`
                <textarea class='performance_question_value' placeholder='Please provide a detailed answer'></textarea>
            `)
        } else if (element["type"] == "likert") {
            main_text_area.append(`
                <div class='performance_question_likert_parent'>
                    Disagree
                    <input class='performance_question_value' type="range" min="1" max="5">
                    Agree
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