import { DEVMODE } from "./globals"
export var UID: string
import { load_data } from './connector'
import { setup_progression } from "./worker_website"
import { range } from "./utils";

globalThis.data = null

const urlParams = new URLSearchParams(window.location.search);
globalThis.uid = urlParams.get('uid');
// take phase from GET if available else use 0 as default
globalThis.phase = parseInt(urlParams.get("phase")) || 0;
globalThis.phaseOverride = parseInt(urlParams.get("phase")) || -1;


function prolific_rewrite_uid(uid) {
    if (uid != "prolific_pilot_1") {
        return uid
    }

    let article = urlParams.get("article") || "any"
    const ALL_ARTICLES = ["classical_chinese_philosophy"]
    if (article == "any") {
        article = ALL_ARTICLES[Math.floor(Math.random() * ALL_ARTICLES.length)]
    }

    let group = urlParams.get("group") || "any"
    const ALL_GROUPS = ["control", "authentic", "generated"]
    if (group == "any") {
        group = ALL_GROUPS[Math.floor(Math.random() * ALL_GROUPS.length)]
    }
    
    globalThis.prolific_pid = urlParams.get('prolific_pid');
    globalThis.session_id = urlParams.get('session_id');
    globalThis.study_id = urlParams.get('study_id');

    return `${article}_${group}`
}

async function get_uid_and_data() {
    // set to "demo" uid if in devmode and uid doesn't exist
    if (DEVMODE && globalThis.uid == null) {
        document.location.href = document.location.href += "?uid=demo_authentic";
    }

    // repeat until we're able to load the data
    while (globalThis.data == null) {
        if (globalThis.uid == null) {
            let UID_maybe = null
            while (UID_maybe == null) {
                UID_maybe = prompt("What is your user id?")
            }
            globalThis.uid = UID_maybe!;
        }

        globalThis.uid = prolific_rewrite_uid(globalThis.uid);

        await load_data().then((data: any) => {
            globalThis.data = data
            globalThis.data_now = globalThis.data[0];
            globalThis.user_control = globalThis.data_now["user_group"] == "control"
            setup_progression()
        }).catch((reason: any) => {
            console.error(reason)
            alert("Invalid UID " + globalThis.uid);
            globalThis.uid = null;
        });
    }

}

get_uid_and_data()
