import { log_data, get_json, get_html, load_data } from "./connector";
import { DEVMODE } from "./globals";
import { range, timer } from "./utils";
import { check_button_lock, instantiate_question, setup_input_listeners, send_signal_to_trigger } from "./worker_utils";

let main_text_area = $("#main_text_area")
let instruction_area_bot = $("#instruction_area_bot")
let instruction_text_bot = $("#instruction_text_bot")
let instruction_area_top = $("#instruction_area_top")

export function setup_progression() {
    function drive_setup() {
        if (globalThis.phase != 0 && globalThis.phaseOverride != globalThis.phase) {
            log_data()
        }

        globalThis.responses = {}
        globalThis.phase_start = new Date().getTime()
        console.log("order_condition",  globalThis.order_condition)

        // preemptively lock
        globalThis.expected_responses = 99999
        check_button_lock()

        send_signal_to_trigger("phase "+globalThis.phase)
        switch (globalThis.phase) {
            case 0:
                setup_intro_information();
                break;
            case 1:
                setup_main_text(0, globalThis.order_condition[0], null);
                break;
            case 2:
                setup_exit_questions();
                break;
            case 3:
                setup_performance_questions();
                break;
            case 4:
                setup_main_text(1, globalThis.order_condition[1], null);
                break;
            case 5:
                setup_exit_questions();
                break;
            case 6:
                setup_performance_questions();
                break;
            case 7:
                setup_main_text(2, globalThis.order_condition[2], null);
                break;
            case 8:
                setup_exit_questions();
                break;
            case 9:
                setup_performance_questions();
                break;
            case 10:
                setup_intro_demographics();
                break;
            case 11:
                load_thankyou();
                break;
        }
    }
    $("#button_next").on("click", () => {
        globalThis.phase += 1;
        drive_setup()
    })

    // fire at the beginning
    drive_setup()

    // setup periodic check for unlocks
    setInterval(check_button_lock, 1000)
}


async function setup_intro_demographics() {
    instruction_area_top.text("Please fill in the following personal information.")
    instruction_text_bot.text("By continuing you agree with the collection and publication of your data for scientific purposes.")

    main_text_area.scrollTop(0)
    main_text_area.html("")

    let questions = await get_json("questions_intro.jsonl")
    globalThis.expected_responses = questions.length

    questions.forEach((question) => {
        main_text_area.append(`${instantiate_question(question)}<br><br>`)
    })
    setup_input_listeners()
}

async function setup_intro_information() {
    instruction_area_top.hide()
    console.log(globalThis.user_control)
    if (globalThis.user_control) {
        main_text_area.html(await get_html("instructions_control.html"))
    } else {
        main_text_area.html(await get_html("instructions.html"))
    }
    globalThis.expected_responses = 0
}

async function setup_main_text(article_id, condition_id, rate_questions: [string, string] | null, rate_questions_intro?: string) {
    // set instructions
    instruction_area_top.show()
    if (rate_questions) {
        instruction_area_top.html(`
            <ul>
                <li>Please evaluate each question (1 - very bad to 5 - very good) shown on the right based on your reading experience.</li>
                ${rate_questions_intro? "<li><b>" + rate_questions_intro + "</b></li>" : ""}
            </ul>
        `)
        // <li>You will be asked if the questions are <b>helpful</b>, <b>distracting</b>, and <b>relevant</b>.</li>
        instruction_text_bot.text("")
    } else {
        instruction_area_top.html(`
            <ul>
                <li>Please read the assigned article. You have <span id="stopwatch">10 minutes ⏱️</span> left.</li>
                <li><b>Click the finished button</b> at the bottom of each paragraph to continue.</li>
                <li>Questions are shown next to the article. When you encounter one, make sure to read and understand it before moving on.</li>
                <li>Think about the questions and keep them in mind as you proceed with reading.</li>
            </ul>
        `)
        instruction_text_bot.text(`
            We will next ask you questions about the article without you being allowed to go back.
        `)

        await timer(10)
        var minutes = 10
        var stopwatchTimer = setInterval(() => {
            minutes -= 1
            if (minutes < 0) {
                minutes = 0
                clearInterval(stopwatchTimer);
            }
            $("#stopwatch").text(`${minutes} minutes ⏱️`)
        }, 1000 * 60);
    }

    await load_data().then((data: any) => {
        globalThis.data = data
        globalThis.data_now = globalThis.data[article_id*3+condition_id];
        globalThis.user_control = globalThis.data_now["user_group"] == "control"
    }).catch((reason: any) => {
        console.error(reason)
        alert("Invalid UID " + globalThis.uid);
        globalThis.uid = null;
    });

    let article = globalThis.data_now["article"]
    // add "finished" button
    if (!rate_questions) {
        article = (
            `<input class="paragraph_finished_button" type='button' value="Start ✅">` +
            article.split("</p>").join(`  <input class="paragraph_finished_button" type='button' value="Finished ✅"></p>`)
        )
    }

    let frame_obj = $(`<div id="article_frame">${article}</div>`)
    let question_obj = $('<div id="main_question_panel"></div>')
    main_text_area.html("")
    main_text_area.append(frame_obj)
    main_text_area.append(question_obj)
    main_text_area.scrollTop(0)

    // hack for JS event loop
    await timer(10)

    if (!rate_questions) {
        let paragraph_offsets = $(".paragraph_finished_button").map((_, element) => $(element).position().top).toArray()
        $(".paragraph_finished_button").each((element_i, element) => {
            if (element_i != paragraph_offsets.length - 1) {
                // make height span two paragraphs to cover question boxes
                let target_height = paragraph_offsets[element_i + 2] - paragraph_offsets[element_i];
                if (element_i + 2 >= paragraph_offsets.length) {
                    target_height = paragraph_offsets[element_i + 1] - paragraph_offsets[element_i]
                }
                frame_obj.append(`<div
                    class="paragraph_blurbox" id="paragraph_blurbox_${element_i}"
                    style="height: ${target_height}px; top: ${paragraph_offsets[element_i] + 30}px; z-index: ${200 - element_i};"
                ></div>`)
            }
            $(element).on("click", () => {
                send_signal_to_trigger("finish reading paragraph "+element_i)
                globalThis.responses[`finish_reading_${element_i}`] = new Date().getTime();
                $(element).css("visibility", "hidden");
                $(`#paragraph_blurbox_${element_i}`).remove()
            })
        })
        // number of paragraphs
        globalThis.expected_responses = paragraph_offsets.length;
    } else {
        globalThis.expected_responses = globalThis.data_now["questions_intext"].length
    }

    // add extra space
    globalThis.data_now["questions_intext"].forEach(async (element, element_i) => {
        let question_span = $(`#question_${element_i}`)
        question_span.append("&nbsp;")
    })

    // hack for JS event loop
    await timer(30)

    globalThis.data_now["questions_intext"].forEach(async (element, element_i) => {
        let paragraph_i = frame_obj.html().split(`id="question_${element_i}"`)[0].split("paragraph_finished_button").length - 1
        let question_span = $(`#question_${element_i}`)
        let offset_x = question_span.position().left
        let offset_y = question_span.position().top

        let question_rate_section = ""
        if (rate_questions) {
            let joined_labels = range(1, 5).map((x) => `<label for="likert_${element_i}_${x}" value="${x}">${x}</label>`).join("\n")
            let joined_inputs = range(1, 5).map((x) => `<input type="radio" name="likert_${element_i}" id="likert_${element_i}_${x}" value="${x}" />`).join("\n")

            question_rate_section = `
                <div class='question_question_likert_parent'>
                    <div class="question_question_likert_labels">${joined_labels}</div>
            
                    ${joined_inputs}<br>

                    <div style="text-align: left;">
                        <span class="question_question_likert_label" style="">${rate_questions[0]}</span>
                        <span class="question_question_likert_label" style="text-align: right; float: right;">${rate_questions[1]}</span>
                    </div>
                </div>
            `
        }

        let offset_y_manual = 0
        if (rate_questions && element_i < globalThis.data_now["questions_intext"].length - 1) {
            let offset_y_next = $(`#question_${element_i + 1}`).position().top
            if (offset_y + 200 >= offset_y_next) {
                offset_y_manual = -40
            }
        }

        question_obj.append(`
            <hr class="arrow" style="width: 0px; height: 8px; position: absolute; left: ${offset_x + 1}px; top: ${offset_y - 8}px;">
            <hr class="line" style="width: ${1000 - offset_x}px; position: absolute; left: ${offset_x + 1}px; top: ${offset_y - 10}px;">
            <div class="question_box" style="position: absolute; top: ${offset_y - 3 + offset_y_manual}px; z-index: ${200 - paragraph_i}">
                <span class="question_box_text">${element["question"]}</span>
                ${question_rate_section}
            </div>
        `)
    })

    setup_input_listeners()
}

async function setup_performance_questions() {
    instruction_area_top.html(`
        <ul>
            <li>Now please answer a few questions about the article.</li>
            <li>Your answers should be exclusively based on the article's content, without any prior knowledge, or external help, such as search engines.</li>
        </ul>
    `)
    instruction_text_bot.html("Once you go to the next page, you are not allowed to change your answer.")
    main_text_area.scrollTop(0)
    main_text_area.html("")

    let questions = globalThis.data_now["questions_performance"]
    globalThis.expected_responses = questions.length

    questions.forEach((question, question_i) => {
        // agreement with Peng that a summary question has this
        let is_summary_question = question["question"].includes("summary_word_count")

        let question_text = question["question"]
        if(is_summary_question) {
            question_text = question_text.replace("summary_word_count", `summary_word_count_q${question_i}`)
        }

        main_text_area.append(`
            <div class="performance_question_text">${question_text}</div>
        `)
        main_text_area.append(`
        <textarea
            class='performance_question_value ${is_summary_question ? "performance_question_100_words" : ""}'
            qid="${question_i}"
            placeholder='Type anything here to activate the go-to-next-screen button'
        ></textarea>
        <br><br>
        `)
        // main_text_area.append(`
        //     <textarea
        //         class='performance_question_value ${is_summary_question ? "performance_question_100_words" : ""}'
        //         qid="${question_i}"
        //         placeholder='Please provide a detailed answer'
        //     ></textarea>
        //     <br><br>
        // `)
    })
    setup_input_listeners()
}

async function setup_exit_questions() {
    instruction_area_top.html(`
        <ul>
            <li>Your answers in this part are vital for our research, so please pay particular attention to your answers.</li>
        </ul>
    `)
    instruction_text_bot.text("")

    main_text_area.scrollTop(0)
    main_text_area.html("")

    let questions = await get_json("questions_exit.jsonl")
    globalThis.expected_responses = questions.filter((question) => !globalThis.user_control || question["also_control"]).length

    let showed_questions = 0
    questions.forEach((question) => {
        // skip this question
        if (globalThis.user_control && !question["also_control"]) {
            return
        }
        showed_questions += 1
        main_text_area.append(`${instantiate_question(question)}<br><br>`)
    })
    setup_input_listeners()

    // continue to the next screen if nothing to show
    if (showed_questions == 0) {
        $("#button_next").trigger("click")
        // hack for JS event loop
        await timer(10)
    }
}

async function load_thankyou() {
    instruction_area_top.hide()
    instruction_area_bot.hide()

    // log last phase
    globalThis.phase += 1;
    log_data()

    main_text_area.html("Please wait 3s for data synchronization to finish.")
    await timer(1000)
    main_text_area.html("Please wait 2s for data synchronization to finish.")
    await timer(1000)
    main_text_area.html("Please wait 1s for data synchronization to finish.")
    await timer(1000)

    let html_text = `Thank you for participating in our study. For any further questions about this project or your data, <a href="mailto:peng.cui@inf.ethz.ch">send us a message</a>.`;
    console.log("PID", globalThis.prolific_pid)
    if (globalThis.prolific_pid != null) {
        html_text += `<br>Please click <a href="https://app.prolific.com/submissions/complete?cc=C693YF4X">this link</a> to go back to Prolific. `
        html_text += `Alternatively use this code <em>C693YF4X</em>.`
    }
    main_text_area.html(html_text);
}