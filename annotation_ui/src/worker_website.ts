import { log_data } from "./connector";
import { DEVMODE } from "./globals";
import { getIndicies } from "./utils";

let main_text_area = $("#main_text_area")


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