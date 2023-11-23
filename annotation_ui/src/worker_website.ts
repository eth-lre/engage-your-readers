import { log_data, get_json, get_html } from "./connector";
import { DEVMODE } from "./globals";
import { range, timer } from "./utils";
import { instantiate_question } from "./worker_utils";

let main_text_area = $("#main_text_area")
let instruction_area_bot = $("#instruction_area_bot")
let instruction_text_bot = $("#instruction_text_bot")
let instruction_area_top = $("#instruction_area_top")

export function setup_progression() {
    function drive_setup() {
        switch (globalThis.phase) {
            case 0: setup_intro_demographics(); break;
            case 1: setup_intro_information(); break;
            case 2: setup_main_text(null); break;
            case 3: setup_performance_questions(); break;
            case 4: setup_exit_questions(); break;
            case 5: 
                if(globalThis.user_control) {
                    // skip to the next one
                    globalThis.phase = 7;
                    drive_setup()
                    return
                }
                setup_main_text("Helpful?");
                break;
            case 6: setup_main_text("Distracting?"); break;
            case 7: load_thankyou(); break;
        }
    }
    $("#but_next").on("click", () => {
        globalThis.phase += 1;
        drive_setup()
    })
    // fire at the beginning
    drive_setup()
}


async function setup_intro_demographics() {
    instruction_area_top.text("Please fill in the following personal information.")
    instruction_text_bot.text("By continuing you agree with the collection and publication of your data for scientific purposes.")

    main_text_area.scrollTop(0)
    main_text_area.html("")

    let questions = await get_json("questions_intro.jsonl")

    questions.forEach((question) => {
        main_text_area.append(`${instantiate_question(question)}<br><br>`)
    })
}

async function setup_intro_information() {
    instruction_area_top.hide()
    main_text_area.html(await get_html("instructions_2.html"))
}

async function setup_main_text(rate_questions: string | null) {
    if (rate_questions) {
        instruction_text_bot.html("Finish reading before continuing.")
    } else {
        instruction_text_bot.html("Please answer all questions before continuing.")
    }

    // set instructions
    instruction_area_top.show()
    if (rate_questions) {
        instruction_area_top.html(`
            <ul>
                <li>Please evaluate the properties of each question (right side).</li>
                <li>TODO more instructions</li>
            </ul>
        `)
    } else {
        instruction_area_top.html(`
            <ul>
                <li>You have <b><span id="stopwatch">20</span> minutes</b> left.</li>
                <li>Make sure to read and comprehend displayed questions before continuing reading.</li>
            </ul>
        `)
        await timer(10)
        var minutes = 20
        var stopwatchTimer = setInterval(() => {
            minutes -= 1
            if (minutes < 0) {
                minutes = 0
                clearInterval(stopwatchTimer);
            }
            $("#stopwatch").text(`${minutes}`)
        }, 1000 * 60);
    }

    let article = globalThis.data_now["article"]
    // add "finished" button
    if (!rate_questions) {
        article = article.split("</p>").join(`  <input class="paragraph_finished_button" type='button' value="Finished"></p>`)
    }

    let frame_obj = $(`<div id="article_frame">${article}</div>`)
    let question_obj = $('<div id="main_question_panel"></div>')
    main_text_area.html("")
    main_text_area.append(frame_obj)
    main_text_area.append(question_obj)
    main_text_area.scrollTop(0)

    // hack for JS event loop
    await timer(10)

    let paragraph_offsets = $(".paragraph_finished_button").map((_, element) => $(element).position().top).toArray()
    $(".paragraph_finished_button").each((element_i, element) => {
        if (element_i != paragraph_offsets.length - 1) {
            // make height span two paragraphs to cover question boxes
            frame_obj.append(`<div
                class="paragraph_blurbox" id="paragraph_blurbox_${element_i}"
                style="height: ${paragraph_offsets[element_i + 2] - paragraph_offsets[element_i]}px; top: ${paragraph_offsets[element_i] + 30}px; z-index: ${200-element_i};"
            ></div>`)
        }
        $(element).on("click", () => {
            element.remove()
            $(`#paragraph_blurbox_${element_i}`).remove()
        })
    })

    globalThis.data_now["questions_intext"].forEach(async (element, element_i) => {
        let paragraph_i = frame_obj.html().split(`id="question_${element_i}"`)[0].split("paragraph_finished_button").length-1
        let question_span = $(`#question_${element_i}`)
        question_span.append("&nbsp;")
        // hack for JS event loop
        await timer(10)

        let offset_x = question_span.position().left
        let offset_y = question_span.position().top

        let question_rate_section = ""
        if (rate_questions) {
            question_rate_section = `
                <div class="question_rate_section">
                    ${rate_questions}&nbsp;&nbsp;
                    <label for="rate_questions_${element_i}_no">No</label>
                    <input name="rate_questions_${element_i}" type="radio" id="rate_questions_${element_i}_no">
                    <input name="rate_questions_${element_i}" type="radio" id="rate_questions_${element_i}_yes">
                    <label for="rate_questions_${element_i}_yes">Yes</label>
                </div>
            `
        }

        question_obj.append(`
            <hr class="arrow" style="width: 0px; height: 8px; position: absolute; left: ${offset_x + 1}px; top: ${offset_y - 8}px;">
            <hr class="line" style="width: ${1000 - offset_x}px; position: absolute; left: ${offset_x + 1}px; top: ${offset_y - 10}px;">
            <div class="question_box" style="position: absolute; top: ${offset_y - 3}px; z-index: ${200-paragraph_i}">
                ${element["question"]}
                ${question_rate_section}
            </div>
        `)
    })
}

async function setup_performance_questions() {
    instruction_area_top.html(`
        <ul>
            <li>Please answer the following questions.</li>
            <li>You are not allowed to go back or use an external tools.</li>
            <li>Base your answers solely on what you just learned and not personal experience.</li>
        </ul>
    `)
    instruction_text_bot.html("Answer all questions before continuing.")
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
    instruction_area_top.html(`
        <ul>
            <li>Please answer the following questions dutifully and with elaboration.</li>
            <li>Make sure that you deliberate before answering each of them and take your time.</li>
        </ul>
    `)

    main_text_area.scrollTop(0)
    main_text_area.html("")

    let questions = await get_json("questions_exit.jsonl")

    questions.forEach((question) => {
        // skip this question
        if(globalThis.user_control && !question["also_control"]) {
            return
        }
        main_text_area.append(`${instantiate_question(question)}<br><br>`)
    })
}

async function load_thankyou() {
    instruction_area_top.hide()
    instruction_area_bot.hide()

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