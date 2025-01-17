INCLUDE /usr/lib/x86_64-linux-gnu/ldscripts/elf_x86_64.x

SECTIONS
{
  .mysection :
  {
    __start_mysection = .;
    KEEP(*(.mysection))
    __stop_mysection = .;
  }
}
