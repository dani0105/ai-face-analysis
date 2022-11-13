fetch("/layouts/custom-navigator/custom-navigator.html")
  .then(stream => stream.text())
  .then(text => define(text));

function define(html) {
  class CustomNavigator extends HTMLElement {

    constructor() {
      super();
      this.innerHTML = html;
      let links = this.getAttribute('links');
      links=JSON.parse(links);

      let link_element = document.createElement('li');
      link_element.className ='nav-link';

      let links_container = this.getElementsByClassName("nav-links")[0]

      for (let index = 0; index < links.length; index++) {
        const element = links[index];
        const clone = link_element.cloneNode(true);
        const a_tag = document.createElement('a');
        a_tag.href = element.link
        a_tag.innerText = element.name
        clone.appendChild(a_tag)
        links_container.appendChild(clone)
      }

    }
  }
  customElements.define('custom-navigator', CustomNavigator);
}